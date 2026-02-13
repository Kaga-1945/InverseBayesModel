using Random

@inline function rand_argmax_isclose(P::AbstractVector{T}, rng::AbstractRNG) where {T<:Real}
    # 最大の確信度を取得
    m = maximum(P)
    # まず候補数を数える（1パス）
    k = 0
    # 最大の確信度に近い値をカウントする
    @inbounds for i in eachindex(P)
        if P[i] ≈ m
            k += 1
        end
    end
    # 候補のうち r 番目を選ぶ（2パス）
    r = rand(rng, 1:k)
    c = 0
    @inbounds for i in eachindex(P)
        if P[i] ≈ m
            c += 1
            if c == r
                return i
            end
        end
    end
    return firstindex(P)  # 到達しない想定
end

@inline function fill_loglik!(P::Vector{Float64}, mu::Vector{Float64}, theta::Float64, tau::Float64)
    inv2tau = 1.0 / (2.0 * tau)
    @inbounds @simd for i in eachindex(mu)
        d = mu[i] - theta
        P[i] = exp(-(d * d) * inv2tau)
    end
    return P
end

abstract type AbstractBayesModel end

mutable struct InverseBayesV1 <: AbstractBayesModel
    # scalars
    tau::Float64
    initial_sigma::Float64
    sigma::Float64
    theta::Float64
    beta::Float64
    alpha::Float64

    # grid
    mu::Vector{Float64}
    max_mu::Int

    # work buffer (確信度ベクトルを毎回確保しない)
    Pbuf::Vector{Float64}

    # histories
    tau_hist::Vector{Float64}
    theta_hist::Vector{Float64}
    alpha_hist::Vector{Float64}
    sigma_hist::Vector{Float64}
    max_mu_hist::Vector{Float64}
end

mutable struct InverseBayesV2 <: AbstractBayesModel
    tau::Float64
    initial_sigma::Float64
    sigma::Float64
    theta::Float64
    beta::Float64
    alpha::Float64
    lambda1::Float64
    lambda2::Float64
    t::Float64

    # histories
    tau_hist::Vector{Float64}
    theta_hist::Vector{Float64}
    alpha_hist::Vector{Float64}
    sigma_hist::Vector{Float64}
    beta_hist::Vector{Float64}
    t_hist::Vector{Float64}

end

function InverseBayesV1(tau::Real, sigma::Real, theta::Real, beta::Real; bin::Int=50, rng::AbstractRNG=Random.default_rng())
    # 平均パラメータの信念分布の分散
    τ = Float64(tau)
    # 事前分散
    Σ0 = Float64(sigma)
    # 事後分散
    Σ = Float64(sigma)
    # 平均パラメータ
    θ = Float64(theta)
    # ベイズのパラメータ
    β = Float64(beta)
    α = 0.0

    mu = collect(range(-2.5, 2.5; length=bin))
    Pbuf = Vector{Float64}(undef, bin)

    fill_loglik!(Pbuf, mu, θ, τ)
    # 確信度が最大のインデックスを保存
    maxidx = rand_argmax_isclose(Pbuf, rng)

    return InverseBayesV1(
        τ, Σ0, Σ, θ, β, α,
        mu, maxidx,
        Pbuf,
        [τ], [θ], [α], [Σ], [mu[maxidx]]
    )
end

function InverseBayesV2(tau::Real, sigma::Real, theta::Real, beta::Real, lambda1::Real, lambda2::Real)
    # 平均パラメータの信念分布の分散
    τ = Float64(tau)
    # 事前分散
    Σ0 = Float64(sigma)
    # 事後分散
    Σ = Float64(sigma)
    # 平均パラメータ
    θ = Float64(theta)
    # ベイズのパラメータ
    β = Float64(beta)
    α = 0.0
    # βの更新規則
    λ₁ = Float64(lambda1)
    λ₂ = Float64(lambda2)
    t = 0.0

    return InverseBayesV2(
        τ, Σ0, Σ, θ, β, α, λ₁, λ₂, t,
        [τ], [θ], [α], [Σ], [β], [t]
    )
end

"""
離散型更新
"""
function update!(m::InverseBayesV1, d::Real; rng::AbstractRNG=Random.default_rng())
    # 観測値
    d = Float64(d)

    # 学習率の更新
    m.alpha = m.tau / (m.tau + (1.0 - m.beta) * m.sigma)

    # ベイズ更新
    tmp_tau = m.alpha * m.sigma
    m.theta = (1.0 - m.alpha) * m.theta + m.alpha * d

    # 逆ベイズ更新
    m.sigma = ((m.sigma + m.tau) / ((1.0 - m.beta) * m.sigma + m.tau)) * m.sigma

    # tauの更新
    m.tau = tmp_tau

    # 確信度計算（正規化省略）
    fill_loglik!(m.Pbuf, m.mu, m.theta, m.tau)
    tmp_max = rand_argmax_isclose(m.Pbuf, rng)

    # 最大確信度のチェック
    if m.max_mu ≠ tmp_max
        m.sigma = m.initial_sigma
        m.max_mu = tmp_max
    elseif m.sigma > 10^10
        m.sigma = m.initial_sigma
    end

    # 履歴保存
    push!(m.tau_hist, m.tau)
    push!(m.theta_hist, m.theta)
    push!(m.alpha_hist, m.alpha)
    push!(m.sigma_hist, m.sigma)
    push!(m.max_mu_hist, m.max_mu)

    return m
end

function update!(m::InverseBayesV2, d::Real)
    d = Float64(d)

    e = (d - m.theta)^2

    # 学習率の更新
    m.alpha = m.tau / (m.tau + (1.0 - m.beta) * m.sigma)

    # ベイズ更新
    tmp_tau = m.alpha * m.sigma
    m.theta = (1.0 - m.alpha) * m.theta + m.alpha * d

    # 逆ベイズ更新
    m.sigma = ((m.sigma + m.tau) / ((1.0 - m.beta) * m.sigma + m.tau)) * m.sigma

    # tauの更新
    m.tau = tmp_tau

    # βの更新
    m.beta = max(0, (1 - m.lambda2) * m.beta + m.lambda2 * ((e - 0.5) / (abs(e - 0.5) + 0.5)))

    if m.sigma > 1e20
        m.sigma = m.initial_sigma
    end

    # 履歴保存
    push!(m.tau_hist, m.tau)
    push!(m.theta_hist, m.theta)
    push!(m.alpha_hist, m.alpha)
    push!(m.sigma_hist, m.sigma)
    push!(m.beta_hist, m.beta)
    push!(m.t_hist, m.t)

end

