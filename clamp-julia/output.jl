using Printf
import JSON
using XML
include("./engine.jl")
using Engine: GreedyCluster
include("./utils.jl")
using Utils: trim_motif, xlogx

clamp_dir = dirname(@__DIR__)
glyph_data = open("$clamp_dir/logo_symbols/glyphs.json", "r") do io
    JSON.parse(io)
end
symbol_library = open("$clamp_dir/logo_symbols/symbol_library.json", "r") do io
    JSON.parse(io)
end

module Output

function write_aligned_transfac(cluster::GreedyCluster, filename::String)
    open(filename, "w") do io
        for i in eachindex(cluster.items)
            motif_id = cluster.items[i].source[1]
            @printf io "AC\t%s\n" motif_id
            @printf io "XX\n"
            @printf io "ID\t%s\n" motif_id
            @printf io "P0\tA\tC\tG\tT\n"
            pfm = cluster.aligned_pfms[i, :, :]
            for j in axes(pfm, 1)
                @printf io "%02d\t%06f\t%06f\t%06f\t%06f\n" j pfm[j, 1] pfm[j, 2] pfm[j, 3] pfm[j, 4]
            end
            @printf io "XX\n//\n"
        end
    end
end

function write_consensus_transfac(cluster::GreedyCluster, filename::String; info_thresh::Float64 = .5)
    c = cluster.idx
    trimmed_pfm, _, _, _ = trim_motif(cluster.aligned_pfms, info_thresh)
    open(filename, "w") do io
        @printf io "AC\t%s%d\n" "cluster" c
        @printf io "XX\n"
        @printf io "ID\t%s%d\n" "cluster" c
        @printf io "P0\tA\tC\tG\tT\n"
        for j in axes(trimmed_pfm, 1)
            @printf io "%02d\t%06f\t%06f\t%06f\t%06f\n" j trimmed_pfm[j, 1] trimmed_pfm[j, 2] trimmed_pfm[j, 3] trimmed_pfm[j, 4]
        end
        @printf io "XX\n//\n"
    end
end

function write_consensus_meme(cluster::GreedyCluster, filename::String; info_thresh::Float64 = .5)
    c = cluster.idx
    trimmed_pfm, _, _, _ = trim_motif(cluster.aligned_pfms, info_thresh)
    open(filename, "w") do io
        @printf io "MEME version 5\n"
        @printf io "ALPHABET= ACGT\n"
        @printf io "MOTIF %s%d %s%d\n" "cluster" c "cluster" c
        @printf io "letter-probability matrix: alength= 4 w= %d nsites= %d E= 0\n" size(trimmed_pfm, 1) sum(item.source[2] for item in cluster.items)
        for j in axes(trimmed_pfm, 1)
            @printf io "%f %f %f %f\n" trimmed_pfm[j, 1] trimmed_pfm[j, 2] trimmed_pfm[j, 3] trimmed_pfm[j, 4]
        end
    end
end

symbol_sets = Dict{Symbol, Tuple{Float64, Vector{Tuple{String, String}}}}()
symbol_sets[:DNA] = (
    SymbolLibrary["DNA"]["max_bits"],
    [(glyph_data[s["name"]]["path"], s["color"]) for s in symbol_library["DNA"]["symbols"]]
)
symbol_sets[:RNA] = (
    SymbolLibrary["RNA"]["max_bits"],
    [(glyph_data[s["name"]]["path"], s["color"]) for s in symbol_library["RNA"]["symbols"]]
)
symbol_sets[:AA] = (
    SymbolLibrary["AA"]["max_bits"],
    [(glyph_data[s["name"]]["path"], s["color"]) for s in symbol_library["AA"]["symbols"]]
)

function plot_logo_stack(aligned_pfms::Array{Float, 3}; symbol::Symbol = :DNA, glyph_width::Float64 = 100., stack_height::Float64 = 200.)::XML.Element
    height, width, _ = size(aligned_pfms)
    svg = XML.Element("svg"; baseProfile="full", version="1.1", xmlns="http://www.w3.org/2000/svg", viewBox="0 0 $(width * glyph_width) $(height * stack_height)")

    logo_stack = XML.Element("g")
    push!(svg, logo_stack)

    for y in 1:height
        pfm = aligned_pfms[y, :, :]
        row = XML.Element("g")
        push!(logo_stack, row)
        for i in 1:width
            pwv = pfm[i, :]
            n = sum(pwv)
            if n == 0
                continue
            end
            vec = xlogx.(pwv ./ n)
            bits = sum(vec) + symbol_sets[symbol][1]
            heights = pwv ./ n .* bits ./ symbol_sets[symbol][1] .* stack_height
            idx = sortperm(pwv)

            stack = XML.Element("g"; transform="translate($((i - 1) * glyph_width))")
            push!(row, stack)

            y_offset = 0.
            for j in idx
                if heights[j] > 0.
                    base = symbol_sets[symbol][2][j]
                    y_offset += heights[j]
                    glyph = XML.Element("path"; d=base[1], fill=base[2], transform="matrix($(glyph_width / 100.) 0 0 $(heights[j] / 100.) 0 $(stack_height - y_offset))")
                    push!(stack, glyph)
                end
            end
        end

        return svg
    end
end

function write_bed_file(cluster, filename)
    open(filename, "w") do io
        for ((chrom, start, stop, strand), sources) in cluster.sites
            @printf io "%s\t%d\t%d\t%s\t.\t%s\n" chrom start stop join((source[0] for source in sources), ";"), strand
        end
    end
end

end