#=
シミュレーションの結果を保存するための下請け関数群
=#

include("Models.jl")
using Plots
gr()

function _savefig(p, model_name::String, condition::String, filename::String)
    save_dir = joinpath(folder, "results", condition, model_name)
    mkpath(save_dir)
    save_path = joinpath(save_dir, filename)
    savefig(p, save_path)
    display(p)
    return p
end

function plot_core_hist(
    m::AbstractBayesModel,
    μ_hist::Vector,
    σ_hist::Vector,
    model_name::String,
    condition::String
)
    # 各系列の長さズレで落ちないように、図ごとに t と L を決める

    t = 1:length(m.P_hist)
    p1 = plot(t, m.P_hist, label="P", xlabel="time", title="P_hist", grid=true, size=(1200, 600))
    _savefig(p1, model_name, condition, "$(model_name)_P_hist.png")

    tx = 1:length(m.x_hist)
    p2 = plot(tx, m.x_hist, label="x", xlabel="time", title="x_hist", grid=true, size=(1200, 600))

    L = min(length(tx), length(μ_hist), length(σ_hist))
    plot!(p2, tx[1:L], μ_hist[1:L], label="μ")

    upper = μ_hist[1:L] .+ σ_hist[1:L]
    lower = μ_hist[1:L] .- σ_hist[1:L]

    plot!(
        p2,
        tx[1:L], upper;
        fillrange=lower,
        fillalpha=0.15,
        linealpha=0.0,
        color=:blue,
        label="σ"
    )

    _savefig(p2, model_name, condition, "$(model_name)_x_hist.png")

    tK = 1:length(m.K_hist)
    p3 = plot(tK, m.K_hist, label="K", xlabel="time", title="K_hist", grid=true, size=(1200, 600))
    _savefig(p3, model_name, condition, "$(model_name)_K_hist.png")

    tR = 1:length(m.R_hist)
    p4 = plot(tR, m.R_hist, label="R", xlabel="time", title="R_hist", grid=true, size=(1200, 600))
    _savefig(p4, model_name, condition, "$(model_name)_R_hist.png")

    return nothing
end

# 追加プロットは「引数列を統一」して多重ディスパッチが確実に動くようにする
plot_extra_hist(::AbstractBayesModel, ::Vector, ::String, ::String) = nothing

function plot_extra_hist(m::InverseBayesV2, σ_hist::Vector, model_name::String, condition::String)
    t = 1:length(m.β_hist)
    p = plot(t, m.β_hist, label="β", xlabel="time", title="β_hist", grid=true, size=(1200, 600))
    _savefig(p, model_name, condition, "$(model_name)_beta_hist.png")
    return nothing
end

function plot_extra_hist(m::InverseBayesV3, σ_hist::Vector, model_name::String, condition::String)
    tβ = 1:length(m.β_hist)
    pβ = plot(tβ, m.β_hist, label="β", xlabel="time", title="β_hist", grid=true, size=(1200, 600))
    _savefig(pβ, model_name, condition, "$(model_name)_beta_hist.png")

    tτ = 1:length(m.τ_hist)
    pτ = plot(tτ, m.τ_hist, label="τ", xlabel="time", title="τ_hist", grid=true, size=(1200, 600))

    L = min(length(tτ), length(σ_hist))
    plot!(pτ, tτ[1:L], σ_hist[1:L], label="σ")

    _savefig(pτ, model_name, condition, "$(model_name)_tau_hist.png")
    return nothing
end

function plot_all_hist(m::AbstractBayesModel, μ_hist::Vector, σ_hist::Vector, model_name::String, condition::String)
    plot_core_hist(m, μ_hist, σ_hist, model_name, condition)
    plot_extra_hist(m, σ_hist, model_name, condition)
    return nothing
end
