using Random

#=
モデルの定義の部分
V1:β固定
V2:τ固定, β変動
v3:
=#

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
    P::Float64
    R₀::Float64
    R::Float64
    x::Float64
    β::Float64
    K::Float64

    # grid
    μ::Vector{Float64}
    max_μ::Int

    # work buffer (確信度ベクトルを毎回確保しない)
    Pbuf::Vector{Float64}

    # histories
    P_hist::Vector{Float64}
    x_hist::Vector{Float64}
    K_hist::Vector{Float64}
    R_hist::Vector{Float64}
    max_μ_hist::Vector{Float64}
end

mutable struct InverseBayesV2 <: AbstractBayesModel
    P::Float64
    R₀::Float64
    R::Float64
    x::Float64
    β::Float64
    K::Float64
    λ::Float64

    # histories
    P_hist::Vector{Float64}
    x_hist::Vector{Float64}
    K_hist::Vector{Float64}
    R_hist::Vector{Float64}
    β_hist::Vector{Float64}

end

mutable struct InverseBayesV3 <: AbstractBayesModel
    P::Float64
    R₀::Float64
    R::Float64
    x::Float64
    β::Float64
    K::Float64
    λ₁::Float64
    λ₂::Float64
    τ::Float64
    c::Float64
    k::Float64

    # histories
    P_hist::Vector{Float64}
    x_hist::Vector{Float64}
    K_hist::Vector{Float64}
    R_hist::Vector{Float64}
    β_hist::Vector{Float64}
    τ_hist::Vector{Float64}

end

# β固定
function InverseBayesV1(P::Real, R₀::Real, x::Real, β::Real; bin::Int=50, rng::AbstractRNG=Random.default_rng())
    # 平均パラメータの信念分布の分散
    P = Float64(P)
    # 事前分散
    R₀ = Float64(R₀)
    # 事後分散
    R = Float64(R₀)
    # 平均パラメータ
    x = Float64(x)
    # ベイズのパラメータ
    β = Float64(β)
    K = 0.0

    μ = collect(range(-2.5, 2.5; length=bin))
    Pbuf = Vector{Float64}(undef, bin)

    fill_loglik!(Pbuf, μ, x, P)
    # 確信度が最大のインデックスを保存
    maxidx = rand_argmax_isclose(Pbuf, rng)

    return InverseBayesV1(
        P, R₀, R, x, β, K,
        μ, maxidx,
        Pbuf,
        [], [], [], [], []
    )
end

# βを動的更新する
function InverseBayesV2(P::Real, R₀::Real, x::Real, λ::Real)
    # 平均パラメータの信念分布の分散
    P = Float64(P)
    # 事前分散
    R₀ = Float64(R₀)
    # 事後分散
    R = Float64(R₀)
    # 平均パラメータ
    x = Float64(x)
    # ベイズのパラメータ
    β = 0.0
    K = 0.0
    # βの更新規則
    λ = Float64(λ)

    return InverseBayesV2(
        P, R₀, R, x, β, K, λ,
        [], [], [], [], []
    )
end

# βを階層的に動的更新する
function InverseBayesV3(P::Real, R₀::Real, x::Real, β::Real, λ₁::Real, λ₂::Real; c::Real=10^10, k::Real=0.3)
    # 平均パラメータの信念分布の分散
    P = Float64(P)
    # 事前分散
    R₀ = Float64(R₀)
    # 事後分散
    R = Float64(R₀)
    # 平均パラメータ
    x = Float64(x)
    # ベイズのパラメータ
    β = 0.0
    K = 0.0
    # βの更新規則
    λ₁ = Float64(λ₁)
    λ₂ = Float64(λ₂)
    τ = 0.5
    c = Float64(c)
    k = Float64(k)

    return InverseBayesV3(
        P, R₀, R, x, β, K, λ₁, λ₂, τ, c, k,
        [], [], [], [], [], []
    )
end

"""
離散型更新
"""
function update!(m::InverseBayesV1, d::Real; rng::AbstractRNG=Random.default_rng())
    # 観測値
    d = Float64(d)

    # 学習率の更新
    m.K = m.P / (m.P + (1.0 - m.β) * m.R)

    # ベイズ更新
    tmp_P = m.K * m.R
    m.x = (1.0 - m.K) * m.x + m.K * d

    # 逆ベイズ更新
    m.R = ((m.R + m.P) / ((1.0 - m.β) * m.R + m.P)) * m.R

    # tauの更新
    m.P = tmp_P

    # 確信度計算（正規化省略）
    fill_loglik!(m.Pbuf, m.μ, m.x, m.P)
    tmp_max = rand_argmax_isclose(m.Pbuf, rng)

    # 最大確信度のチェック
    if m.max_μ ≠ tmp_max
        m.R = m.R₀
        m.max_μ = tmp_max
    elseif m.R > 10^10
        m.R = m.R₀
    end

    # 履歴保存
    push!(m.P_hist, m.P)
    push!(m.x_hist, m.x)
    push!(m.K_hist, m.K)
    push!(m.R_hist, m.R)
    push!(m.max_μ_hist, m.max_μ)

    return m
end

function update!(m::InverseBayesV2, d::Real)
    d = Float64(d)

    e² = (d - m.x)^2

    # 学習率の更新
    m.K = m.P / (m.P + (1.0 - m.β) * m.R)

    # ベイズ更新
    tmp_P = m.K * m.R
    m.x = (1.0 - m.K) * m.x + m.K * d

    # 逆ベイズ更新
    m.R = ((m.R + m.P) / ((1.0 - m.β) * m.R + m.P)) * m.R

    # tauの更新
    m.P = tmp_P

    # βの更新
    m.β = max(0, (1 - m.λ) * m.β + m.λ * ((e² - 0.1) / (abs(e² - 0.1) + 0.5)))

    if m.R > 1e10
        m.R = m.R₀
    end

    # 履歴保存
    push!(m.P_hist, m.P)
    push!(m.x_hist, m.x)
    push!(m.K_hist, m.K)
    push!(m.R_hist, m.R)
    push!(m.β_hist, m.β)

end

function update!(m::InverseBayesV3, d::Real)
    d = Float64(d)

    # 予測誤差
    e = (d - m.x)
    e² = e^2

    # 学習率の更新
    m.K = m.P / (m.P + (1.0 - m.β) * m.R)

    # ベイズ更新
    # 事後分散の計算
    tmp_P = m.K * m.R

    # 推定値の更新
    #m.x = (1.0 - m.K) * m.x + m.K * d
    m.x = m.x + m.K * e

    #m.R = ((m.R + m.P) / ((1.0 - m.β) * m.R + m.P)) * m.R
    m.R = max(1e-10, ((m.R + m.P) / ((1.0 - m.β) * m.R + m.P)) * m.R - 0.1 * (m.R - 0))
    #m.R = ((m.R + m.P) / ((1.0 - m.β) * m.R + m.P)) * m.R * (1 - m.R / 10^10)

    # 事後分散の更新
    m.P = tmp_P

    # 基準値の更新
    m.τ = (1 - m.λ₁) * m.τ + λ₁ * min(e², m.c * m.τ)

    # βの更新
    m.β = max(0, (1 - m.λ₂) * m.β + m.λ₂ * ((e² - m.τ) / (abs(e² - m.τ) + m.k * m.τ)))

    # 事後分散のリセット
    if m.R > 10^10
        m.R = m.R₀
    end

    # 履歴保存
    push!(m.P_hist, m.P)
    push!(m.x_hist, m.x)
    push!(m.K_hist, m.K)
    push!(m.R_hist, m.R)
    push!(m.β_hist, m.β)
    push!(m.τ_hist, m.τ)

end

