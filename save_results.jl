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

function plot_core_hist(m::AbstractBayesModel, μ_hist::Vector, model_name::String, condition::String)
    t = 1:length(m.P_hist)

    p1 = plot(t, m.P_hist, label="P", xlabel="time", title="P_hist", grid=true, size=(1200, 600))
    _savefig(p1, model_name, condition, "$(model_name)_P_hist.png")

    p2 = plot(t, m.x_hist, label="x", xlabel="time", title="x_hist", grid=true, size=(1200, 600))
    plot!(p2, t, μ_hist, label="μ")
    _savefig(p2, model_name, condition, "$(model_name)_x_hist.png")

    p3 = plot(t, m.K_hist, label="K", xlabel="time", title="K_hist", grid=true, size=(1200, 600))
    _savefig(p3, model_name, condition, "$(model_name)_K_hist.png")

    p4 = plot(t, m.R_hist, label="R", xlabel="time", title="R_hist", grid=true, size=(1200, 600))
    _savefig(p4, model_name, condition, "$(model_name)_R_hist.png")

    return nothing
end

plot_extra_hist(::AbstractBayesModel, ::String, ::String) = nothing

function plot_extra_hist(m::InverseBayesV2, model_name::String, condition::String)
    t = 1:length(m.β_hist)
    p = plot(t, m.β_hist, label="β", xlabel="time", title="β_hist", grid=true, size=(1200, 600))
    _savefig(p, model_name, condition, "$(model_name)_beta_hist.png")
    return nothing
end

function plot_extra_hist(m::InverseBayesV3, model_name::String, condition::String)
    tβ = 1:length(m.β_hist)
    pβ = plot(tβ, m.β_hist, label="β", xlabel="time", title="β_hist", grid=true, size=(1200, 600))
    _savefig(pβ, model_name, condition, "$(model_name)_beta_hist.png")

    tτ = 1:length(m.τ_hist)
    pτ = plot(tτ, m.τ_hist, label="τ", xlabel="time", title="τ_hist", grid=true, size=(1200, 600))
    _savefig(pτ, model_name, condition, "$(model_name)_tau_hist.png")

    return nothing
end

function plot_all_hist(m::AbstractBayesModel, μ_hist::Vector, model_name::String, condition::String)
    plot_core_hist(m, μ_hist, model_name, condition)
    plot_extra_hist(m, model_name, condition)  # ← 型で自動的にV1/V2/V3が選ばれる
    return nothing
end
