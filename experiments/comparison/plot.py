import numpy as np
import pandas as pd

import qmf


# Set directories
data = "kodak"
results_dir = "experiments/comparison"
figures_dir = (
    "/scratch/Dropbox/Apps/Overleaf/qmf/v4-information_sciences/manuscript/figures"
)


# Load results
results = qmf.read_config(f"{results_dir}/{data}_results.json")
results = pd.DataFrame(results)
results = results.query("`bit rate (bpp)` < 0.8")

# Plot PSNR vs bpp
plot = qmf.Plot(results, columns=("data", "method", "bit rate (bpp)", "PSNR (dB)"))

plot.interpolate(
    x="bit rate (bpp)",
    y="PSNR (dB)",
    x_values=np.linspace(0.05, 0.5, 19),
    groupby=["method", "data"],
)

plot.plot(
    x="bit rate (bpp)",
    y="PSNR (dB)",
    groupby="method",
    errorbar="se",
    dashed=True,
    xlim=(0.05, 0.5),
    ylim=(17, None),
    legend_labels=["JPEG", "SVD", "QMF"],
)

plot.save(save_dir=results_dir, prefix=data, format="pdf")
plot.save(save_dir=figures_dir, prefix=data, format="pgf")


# Plot MS-SSIM vs bpp
plot = qmf.Plot(results, columns=("data", "method", "bit rate (bpp)", "MS-SSIM"))

plot.interpolate(
    x="bit rate (bpp)",
    y="MS-SSIM",
    x_values=np.linspace(0.05, 0.5, 19),
    groupby=["method", "data"],
)

plot.plot(
    x="bit rate (bpp)",
    y="MS-SSIM",
    groupby="method",
    errorbar="se",
    dashed=True,
    xlim=(0.05, 0.5),
    ylim=(0.675, None),
    legend_labels=["JPEG", "SVD", "QMF"],
)

plot.save(save_dir=results_dir, prefix=data, format="pdf")
plot.save(save_dir=figures_dir, prefix=data, format="pgf")


# Plot encoding time vs bpp
plot = qmf.Plot(
    results, columns=("data", "method", "bit rate (bpp)", "encoding time (ms)")
)

plot.interpolate(
    x="bit rate (bpp)",
    y="encoding time (ms)",
    x_values=np.linspace(0.05, 0.5, 19),
    groupby=["method", "data"],
)

plot.plot(
    x="bit rate (bpp)",
    y="encoding time (ms)",
    groupby="method",
    errorbar="se",
    dashed=True,
    legend_labels=["JPEG", "SVD", "QMF"],
)

plot.save(save_dir=results_dir, prefix=data, format="pdf")


# Plot decoding time vs bpp
plot = qmf.Plot(
    results, columns=("data", "method", "bit rate (bpp)", "decoding time (ms)")
)

plot.interpolate(
    x="bit rate (bpp)",
    y="decoding time (ms)",
    x_values=np.linspace(0.05, 0.5, 19),
    groupby=["method", "data"],
)

plot.plot(
    x="bit rate (bpp)",
    y="decoding time (ms)",
    groupby="method",
    errorbar="se",
    dashed=True,
    legend_labels=["JPEG", "SVD", "QMF"],
)

plot.save(save_dir=results_dir, prefix=data, format="pdf")
