include("Simulation_env.jl")
include("Models.jl")
include("save_results.jl")
using Random
using Plots

#= シミュレーションのパラメータ設定 =#
TIME = 10000
P = 0.5
R = 1.0
x = 0.0
β = 0.05
bin = 50
λ₁ = 0.001
λ₂ = 0.05

μᵤ = 2.5
μₗ = -2.5
σ = 0.3


folder = "figures"

function main()
    # 乱数生成器を作成
    rng = MersenneTwister(123)
    # 逆ベイズモデルの初期化
    #model = InverseBayesV1(P, R, x, β; bin=bin, rng=rng)
    #model = InverseBayesV2(P, R, x, λ₁)
    model = InverseBayesV3(P, R, x, β, λ₁, λ₂)

    model_name = string(nameof(typeof(model)))
    condition = "Unsteady"
    # データを生成
    samples_s, loc_line_s, scale_line = generate_unsteady_series(μₗ, μᵤ, σ, TIME, SEED=1919)

    # シミュレーション 
    @inbounds @simd for d in samples_s
        #update!(model, d; rng=rng)
        update!(model, d)
    end

    # 推定値の推移をプロットする
    plot_all_hist(model, loc_line_s, scale_line, string(typeof(model)), condition)

end

main()