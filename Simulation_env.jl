# シミュレーション生成器（Julia版・割り当てを抑えた実装）
# 依存: Random（標準）
using Random

"""
    generate_steady_series(loc, scale, size; rng=Random.default_rng())

定常環境の生成
- loc   : 平均
- scale: 標準偏差
- size  : サンプル数

返り値: (samples, loc_line)
"""
function generate_steady_series(loc::Real, scale::Real, size::Integer; rng::AbstractRNG=Random.default_rng(), SEED::Int64=123)
    Random.seed!(SEED)
    # 平均
    μ = Float64(loc)
    # 標準偏差
    σ = Float64(scale)
    # サンプル数
    n = Int(size)

    # undef ~ 初期化を省いた実装。意味不明な値が入るが書き換えるから良い
    samples = Vector{Float64}(undef, n)
    # 高速のfor文
    @inbounds @simd for i in 1:n
        samples[i] = μ + σ * randn(rng)
    end

    loc_line = fill(μ, n)  # loc を size 個並べる
    return samples, loc_line
end

"""
    generate_unsteady_series(upper_loc, lower_loc, scale, size; rng=Random.default_rng())

非定常環境の生成
- upper_loc : 平均の上限
- lower_loc : 平均の下限
- scale     : 標準偏差
- size      : 全体のサンプル数

仕様（Python版と同じ）:
- loc は最初に一様乱数で [lower_loc, upper_loc] から選ぶ
- 各時点 i で、rand() < 0.001 かつ t > 500 のとき loc を再サンプルし、priod に i を追加し、t を 0 に戻す
- その loc を平均として正規乱数を1つ生成

返り値: (samples, loc_line, priod)
- priod は loc が切り替わった時点のインデックス（1始まり）
"""
function generate_unsteady_series(upper_loc::Real, lower_loc::Real, scale::Real, size::Integer;
    rng::AbstractRNG=Random.default_rng(), SEED::Int64=123)
    Random.seed!(SEED)
    upper = Float64(upper_loc)
    lower = Float64(lower_loc)
    σ = Float64(scale)
    n = Int(size)

    samples = Vector{Float64}(undef, n)
    loc_line = Vector{Float64}(undef, n)
    priod = Int[]  # 変更点のインデックスを格納

    t = 0
    loc = lower + (upper - lower) * rand(rng)

    @inbounds for i in 1:n
        # 0.1%で平均を変更させる
        if rand(rng) < 0.001 && t > 500
            # 平均値を更新
            loc = lower + (upper - lower) * rand(rng)
            # インデックスを記録する
            push!(priod, i)
            # カウントをリセット
            t = 0
        end
        samples[i] = loc + σ * randn(rng)
        loc_line[i] = loc
        t += 1
    end

    return samples, loc_line, priod
end
