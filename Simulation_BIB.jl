include("Simulation_env.jl")
include("Inverse_Bayse.jl")
using Random
using Plots

TIME = 10000
P = 0.5
R = 1.0
x = 0.0
β = 0.0
bin = 50
λ₁ = 0.001
λ₂ = 0.1

μ = 0.0
σ = 0.3

folder = "figures"

function plot_theta(theta_hist::Array, μ_hist::Array, model_name::String, condition::String)
    #=
    環境に設定した平均値μとモデルの推定値を描画する関数
    =#
    times = 1:TIME  # x軸（0〜10000）
    xt = 1:TIME/10:TIME

    p = plot(
        times, μ_hist,
        color=:blue,
        linewidth=1,
        label="μ",
        xlabel="time",
        ylabel="Value",
        title="Steady environment",
        grid=true,
        size=(1200, 600),
    )

    plot!(
        p,
        times, theta_hist[2:end],
        color=:orange,
        linewidth=1,
        label="θ",
    )

    xlims!(p, 0, TIME)
    ylims!(p, -1, 1)
    xticks!(p, xt)

    plot!(
        p,
        tickfontsize=12,
        guidefontsize=14,
        legendfontsize=14,
        titlefontsize=16,
    )

    # 画像の保存
    save_dir = joinpath(folder, "results", "θ_trajectory", condition, model_name)
    mkpath(save_dir)
    filename = "$(model_name)_θ_trajectory.png"
    save_path = joinpath(save_dir, filename)
    savefig(p, save_path)

    display(p)
end

function main()
    # 乱数生成器を作成
    rng = MersenneTwister(123)
    # 逆ベイズモデルの初期化
    model = InverseBayesV1(P, R, x, β; bin=bin, rng=rng)
    #model = InverseBayesV2(τ, Σ, θ, β, λ₁, λ₂)

    model_name = string(nameof(typeof(model)))
    condition = "Steady"
    # データを生成
    samples_s, loc_line_s = generate_steady_series(μ, σ, TIME)

    # シミュレーション 
    @inbounds @simd for d in samples_s
        #update!(model, d; rng=rng)
        update!(model, d)
    end

    # 推定値の推移をプロットする
    plot_theta(model.x_hist, loc_line_s, model_name, condition)

end

main()