include("./engine.jl")

module Input

function parse_meme_files(meme_files::Vector{String}; get_sites::Bool = true,
    nsites_pattern::Regex = r"letter-probability matrix: alength= \d+ w= \d+ nsites= (\d+) E= ([\d.+e-]+)",
    weights_pattern::Regex = r"\s*([\d\.]+)\s*([\d\.]+)\s*([\d\.]+)\s*([\d\.]+)",
    width_pattern::Regex = r"MOTIF.+width =\s+(\d+)")::Vector{Engine.GreedyItem}

    items = Vector{Engine.GreedyItem}()
    sites = Dict{Int64, Set{Tuple{String, Int64, Int64, Char}}}()
    n = 0
    for fn in meme_files
        file_n::Int64 = 1
        width::Int64 = 0
        open(fn, "r") do io
            while !eof(io)
                line = readline(io)
                width_match = match(width_pattern, line)
                if width_match !== nothing
                    width = parse(Int64, width_match[1])
                end
                if endswith(strip(line), "sites sorted by position p-value") && get_sites
                    readline(io)
                    readline(io)
                    readline(io)
                    line = readline(io)
                    sites[n] = Set{Tuple{String, Int64, Int64, Char}}()
                    while startswith(line, "chr")
                        region = split(strip(line), "(")[1]
                        offset = parse(Int64, line[31:37])
                        chrom, coords = split(region, ":")
                        start = parse(Int64, split(coords, "-")[1]) + offset - 1
                        stop = start + width
                        push!(sites[n], (chrom, start, stop, line[30]))
                        line = readline(io)
                    end
                end

                nsites_match = match(nsites_pattern, line)
                if nsites_match !== nothing
                    nsites = parse(Int64, nsites_match[1])
                    evalue = parse(Float64, nsites_match[2])
                    pfm = Matrix{Float64}(undef, width, 4)
                    line = readline(io)
                    weights_match = match(weights_pattern, line)
                    i = 1
                    while weights_match !== nothing
                        for j in 1:4
                            pfm[i, j] = parse(Float64, weights_match[j]) * nsites
                        end
                        line = readline(io)
                        weights_match = match(weights_pattern, line)
                        i += 1
                    end
                    source = (fn * "-motif" * string(file_n), nsites, evalue)
                    push!(items, Engine.GreedyItem(n, pfm, source,
                        get_sites ? sites[n] : Set{Tuple{String, Int64, Int64, Char}}()))
                    file_n += 1
                    n += 1
                end
            end
        end
    end
    
    return items
end

end