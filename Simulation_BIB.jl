include("Simulation_env.jl")
include("Inverse_Bayse.jl")
using Random
using Plots

TIME = 10000
τ = 0.5
Σ = 1.0
θ = 0.0
β = 0.0
bin = 50

λ₁ = 0.001
λ₂ = 0.1
μ = 0.0
σ = 0.3

folder = "figures"

function main()
    # 乱数生成器を作成
    rng = MersenneTwister(123)
    # 逆ベイズモデルの初期化
    #model = InverseBayesV1(τ, Σ, θ, β; bin=bin, rng=rng)
    model = InverseBayesV2(τ, Σ, θ, β, λ₁, λ₂)
    # データを生成
    samples_s, loc_line_s = generate_steady_series(μ, σ, TIME)

    # シミュレーション 
    @inbounds @simd for d in samples_s
        #update!(model, d; rng=rng)
        update!(model, d)
    end

    times = 1:TIME  # x軸（0〜10000）
    xt = 1:TIME/10:TIME

    p = plot(
        times, loc_line_s,
        color=:blue,
        linewidth=1,
        label="τ",
        xlabel="time",
        ylabel="Value",
        title="Steady environment",
        grid=true,
        size=(1200, 600),
    )

    plot!(
        p,
        times, model.theta_hist[2:end],
        color=:orange,
        linewidth=1,
        label="β",
    )

    xlims!(p, 0, TIME)
    xticks!(p, xt)

    plot!(
        p,
        tickfontsize=12,
        guidefontsize=14,
        legendfontsize=14,
        titlefontsize=16,
    )

    # 画像の保存
    isdir(folder) || mkdir(folder)
    filename = "Steady_Bayse_β:$(model.beta)_$(typeof(model)).png"
    save_path = joinpath(folder, filename)
    savefig(p, save_path)

    display(p)


end

main()