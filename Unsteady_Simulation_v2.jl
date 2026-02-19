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

# τの更新パラメータ
λ₁ = 0.01

# βの更新パラメータ
λ₂ = 0.1

μᵤ = 2.5
μₗ = -2.5
σᵤ = 1.0
σₗ = 0.0



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
    samples_s, loc_line, scale_line = generate_unsteady_series_2(μₗ, μᵤ, σₗ, σᵤ, TIME, SEED=1919)

    # シミュレーション 
    @inbounds @simd for d in samples_s
        #update!(model, d; rng=rng)
        update!(model, d)
    end

    # 推定値の推移をプロットする
    plot_all_hist(model, loc_line, scale_line, string(typeof(model)), condition)

end

main()