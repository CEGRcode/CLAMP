using Statistics

module Utils

function highest_n_info_sum(pfm::Matrix{Float64}; n::Int64 = 4, w::Int64 = 3)::Float64
    pwm = pfm ./ sum(pfm, dims=2)
    pwm[pwm .== 0.] .= 1.
    bits = dropdims(sum(pwm .* log2.(pwm), dims=2), dims=2) .+ log2(size(pfm, 2))
    val = sum(bits[1:w]) / w
    mean_bits = zeros(Float64, size(aligned_pfms, 2) - w + 1)
    mean_bits[1] = val
    for i in w + 1:length(bits)
        val += (bits[i] - bits[i - w]) / w
        mean_bits[i - w + 1] = val
    end
    return sum(sort(mean_bits, rev=true)[1:n])
end

function check_periodicity(pfm::Matrix{Float64}; p::Int64 = 1)::Float64
    pwm = pfm ./ sum(pfm, dims=2)
    pwm[pwm .== 0.] .= 1.
    bits = dropdims(sum(pwm .* log2.(pwm), dims=2), dims=2) .+ log2(size(pfm, 2))
    w = size(pfm, 1)
    corr_sum = 0.
    total_bit_prod = 0.
    for offset in p:p:w - 1
        for i in 1:w - offset
            corr = cor(pfm[i, :], pfm[i + offset, :])
            bit_prod = bits[i] * bits[i + offset]
            if !isnan(corr)
                corr_sum += corr * bit_prod
            end
            total_bit_prod += bit_prod
        end
    end
    return corr_sum / total_bit_prod
end

function boltzmann(arr::Array{Float64}, alpha::Float64, dims::Union{Int64, Tuple})
    return sum(arr .* exp.(alpha .* arr), dims=dims) ./ sum(exp.(alpha .* arr), dims=dims)
end

function trim_motif(aligned_pfms::Array{Float64, 3}; info_thresh::Float64 = .5,
        w::Int64 = 2)::Tuple{Matrix{Float64}, Int64, Int64, Bool}
    pfm = dropdims(sum(aligned_pfms, dims=1), dims=1)

    if size(aligned_pfms, 1) == 1
        return pfm, 0, 0, false
    end

    pwm = pfm ./ sum(pfm, dims=2)
    pwm[pwm .== 0.] .= 1.
    bits = dropdims(sum(pwm .* log2.(pwm), dims=2), dims=2) .+ log2(size(aligned_pfms, 3))
    val = sum(bits[1:w]) / w
    mean_bits = zeros(Float64, size(aligned_pfms, 2) - w + 1)
    mean_bits[1] = val
    for i in w + 1:length(bits)
        val += (bits[i] - bits[i - w]) / w
        mean_bits[i - w + 1] = val
    end

    informative_bits = findall(x -> x > info_thresh, mean_bits)
    if length(informative_bits) <= 1
        return pfm, 0, 0, false
    end
    start = informative_bits[1]
    end_ = informative_bits[end] + w - 1
    return pfm[start:end_, :], start - 1, size(pfm, 1) - end_, true
end

function xlogx(x::Real)::Float64
    return x == 0. ? 0. : x * log(x)
end

end