using Plots
include("Simulation_env.jl")
gr()

# =========================
# 乱数の生成（定常環境）
# =========================
samples_s, loc_line_s = generate_steady_series(0, 0.3, 10_000)
times = 0:9_999   # np.arange(0, 10000) に対応

# =========================
# サンプルの可視化
# =========================
p = plot(
    times,
    loc_line_s,
    color=:blue,
    linewidth=1,
    label="μ",
    xlabel="T",
    ylabel="observed data",
    title="steady enviroment",
    grid=true,
    legend=:topright,
    size=(1200, 600),   # figsize=(12, 6)
)

scatter!(
    p,
    times,
    samples_s,
    color=:orange,
    markersize=2,       # s=2
    markerstrokewidth=0,
    label="observed data",
)

# =========================
# 軸・目盛り設定
# =========================
xlims!(p, 0, 10_000)
xticks!(p, 0:1000:10_000)

plot!(
    p,
    tickfontsize=10,
    guidefontsize=12,
    legendfontsize=12,
    titlefontsize=14,
)

# =========================
# 画像の保存
# =========================
folder = "figures"
isdir(folder) || mkdir(folder)

filename = "Steady_environment.png"
save_path = joinpath(folder, filename)

savefig(p, save_path)

# =========================
# 画像の表示
# =========================
display(p)
