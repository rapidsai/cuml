# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import datetime

import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA


def create_plotly_plots(datasets, embeddings, all_metrics, spectral_inits):
    """Create interactive Plotly plots for web display.

    Parameters
    ----------
    datasets : dict
        Mapping of dataset name to (X, colors).
    embeddings : dict
        Mapping of dataset name to low-dimensional UMAP embedding or dict with 'reference'/'cuml' keys.
    all_metrics : dict
        Mapping of dataset name to metrics dict or dict with 'reference'/'cuml' keys.
    spectral_inits : dict
        Mapping of dataset name to 2-D spectral initialisation or dict with 'reference'/'cuml' keys.

    Returns
    -------
    main_html : list[str]
        Plotly divs for the 4-panel figure per dataset.
    spectral_html : list[str]
        Plotly divs for the spectral-initialisation scatter per dataset.
    comparison_data : dict
        Data structure indicating which datasets have both implementations.
    """

    main_html = []
    spectral_html = []
    comparison_data = {}

    import re

    for name, (X, colors) in datasets.items():
        # Generate safe name for HTML IDs (consistent with JavaScript)
        safe_name = re.sub(r"[^a-zA-Z0-9]", "-", name).lower()

        # Check if this dataset has both implementations (comparison mode)
        is_comparison = (
            isinstance(embeddings[name], dict)
            and "reference" in embeddings[name]
            and "cuml" in embeddings[name]
        )

        if is_comparison:
            # Store comparison data for later use
            comparison_data[name] = {
                "reference": {
                    "embedding": embeddings[name]["reference"],
                    "spectral_init": spectral_inits[name].get("reference"),
                    "metrics": all_metrics[name]["reference"],
                },
                "cuml": {
                    "embedding": embeddings[name]["cuml"],
                    "spectral_init": spectral_inits[name].get("cuml"),
                    "metrics": all_metrics[name]["cuml"],
                },
            }

            # Use reference implementation as default for initial display
            embedding = embeddings[name]["reference"]
            spec_init = spectral_inits[name].get("reference")
            metrics = all_metrics[name]["reference"]
        else:
            # Single implementation mode
            embedding = embeddings[name]
            spec_init = spectral_inits.get(name)
            metrics = all_metrics[name]

        # If the original data has more than 3 features, project to 3-D via PCA for visualisation
        pca_used = False
        if X.shape[1] > 3:
            pca = PCA(n_components=3, random_state=42)
            X_vis = pca.fit_transform(X)
            pca_used = True
        else:
            X_vis = X

        is_3d = X_vis.shape[1] == 3

        # Build subplot figure with dynamic title for the original-data plot
        orig_title = "Original Data (PCA)" if pca_used else "Original Data"

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                orig_title,
                "UMAP Embedding",
                "Quality Metrics",
                "Persistence Diagram",
            ],
            specs=[
                [
                    {"type": "scatter3d" if is_3d else "scatter"},
                    {"type": "scatter"},
                ],
                [{"type": "table"}, {"type": "scatter"}],
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
        )

        # Original data plot
        if is_3d:
            scatter_orig = go.Scatter3d(
                x=X_vis[:, 0],
                y=X_vis[:, 1],
                z=X_vis[:, 2],
                mode="markers",
                marker=dict(
                    size=4,
                    color=colors,
                    colorscale="Turbo",  # vivid perceptually uniform colormap
                    showscale=False,  # hide duplicate colorbar
                ),
                showlegend=False,
                name="Original Data",
                hovertemplate="<b>Original Data</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>",
            )
        else:
            scatter_orig = go.Scatter(
                x=X_vis[:, 0],
                y=X_vis[:, 1],
                mode="markers",
                marker=dict(
                    size=6, color=colors, colorscale="Turbo", showscale=False
                ),
                showlegend=False,
                name="Original Data",
                hovertemplate="<b>Original Data</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>",
            )
        fig.add_trace(scatter_orig, row=1, col=1)

        # UMAP embedding plot
        scatter_umap = go.Scatter(
            x=embedding[:, 0],
            y=embedding[:, 1],
            mode="markers",
            marker=dict(
                size=6,
                color=colors,
                colorscale="Turbo",
                showscale=True,
                colorbar=dict(
                    title=dict(text="Class", side="right"),
                    x=0.48,  # Position closer to the UMAP subplot
                    xanchor="left",
                    y=0.75,  # Position at the UMAP subplot level
                    yanchor="middle",
                    len=0.4,  # Make it smaller to fit better
                    thickness=15,
                ),
            ),
            name="UMAP Embedding",
            showlegend=False,
            hovertemplate="<b>UMAP Embedding</b><br>UMAP1: %{x:.3f}<br>UMAP2: %{y:.3f}<extra></extra>",
        )
        fig.add_trace(scatter_umap, row=1, col=2)

        # Metrics table
        metrics_data = [
            ["Trustworthiness", f"{metrics.get('trustworthiness', 0):.4f}"],
            ["Continuity", f"{metrics.get('continuity', 0):.4f}"],
            [
                "Geodesic Spearman Corr",
                f"{metrics.get('geodesic_spearman_correlation', 0):.4f}",
            ],
            [
                "Geodesic Pearson Corr",
                f"{metrics.get('geodesic_pearson_correlation', 0):.4f}",
            ],
            ["DEMaP", f"{metrics.get('demap', 0):.4f}"],
            [
                "Fuzzy KL (high vs low)",
                f"{metrics.get('fuzzy_kl_divergence', 0):.4f}",
            ],
            [
                "Fuzzy Sym KL (high vs low)",
                f"{metrics.get('fuzzy_sym_kl_divergence', 0):.4f}",
            ],
            [
                "Betti H0 (high→low)",
                f"{metrics.get('betti_h0_high', 'N/A')} → {metrics.get('betti_h0_low', 'N/A')}",
            ],
            [
                "Betti H1 (high→low)",
                f"{metrics.get('betti_h1_high', 'N/A')} → {metrics.get('betti_h1_low', 'N/A')}",
            ],
        ]

        # Add comparison metrics if available
        if "avg_knn_recall" in metrics:
            metrics_data.append(
                ["Avg KNN Recall (vs Ref)", f"{metrics['avg_knn_recall']:.4f}"]
            )
        if "mae_knn_dist" in metrics:
            metrics_data.append(
                ["KNN Distance MAE (vs Ref)", f"{metrics['mae_knn_dist']:.4f}"]
            )
        if "kl_sym" in metrics:
            metrics_data.append(
                [
                    "Fuzzy Graph Sym KL (ref vs cuML)",
                    f"{metrics['kl_sym']:.4f}",
                ]
            )
        if "jacc" in metrics:
            metrics_data.append(
                [
                    "Fuzzy Graph Edge Jaccard (ref vs cuML)",
                    f"{metrics['jacc']:.4f}",
                ]
            )
        if "row_l1" in metrics:
            metrics_data.append(
                [
                    "Fuzzy Graph Row-sum L1 (ref vs cuML)",
                    f"{metrics['row_l1']:.4f}",
                ]
            )

        table = go.Table(
            header=dict(
                values=["Metric", "Value"],
                fill_color="lightblue",
                align="left",
                font_size=12,
                height=30,
            ),
            cells=dict(
                values=list(zip(*metrics_data)),
                fill_color="white",
                align="left",
                font_size=11,
                height=25,
            ),
        )
        fig.add_trace(table, row=2, col=1)

        # Persistence diagram plot
        pd_available = "high_pd" in metrics and "low_pd" in metrics
        if pd_available:
            try:
                import numpy as np

                # Helper to safely extract diagrams and convert to ndarray
                def _extract(dgms, idx):
                    if len(dgms) > idx and len(dgms[idx]):
                        return np.asarray(dgms[idx])
                    return np.empty((0, 2))

                high_pd_full = metrics["high_pd"]
                low_pd_full = metrics["low_pd"]

                high_h0 = _extract(high_pd_full, 0)
                high_h1 = _extract(high_pd_full, 1)
                low_h0 = _extract(low_pd_full, 0)
                low_h1 = _extract(low_pd_full, 1)

                # Remove infinite deaths in H0 diagrams (the infinite bar)
                high_h0 = (
                    high_h0[np.isfinite(high_h0[:, 1])]
                    if high_h0.size
                    else high_h0
                )
                low_h0 = (
                    low_h0[np.isfinite(low_h0[:, 1])]
                    if low_h0.size
                    else low_h0
                )

                # Plotting helper
                def _plot(points, color, symbol, name):
                    if points.size:
                        fig.add_trace(
                            go.Scatter(
                                x=points[:, 0],
                                y=points[:, 1],
                                mode="markers",
                                marker=dict(
                                    size=6, color=color, symbol=symbol
                                ),
                                name=name,
                            ),
                            row=2,
                            col=2,
                        )

                # Add four diagrams with distinct colours
                _plot(high_h0, "#1f77b4", "circle", "High-dim H0")  # blue
                _plot(high_h1, "#2ca02c", "diamond", "High-dim H1")  # green
                _plot(low_h0, "#d62728", "circle", "Low-dim H0")  # red
                _plot(low_h1, "#ff7f0e", "diamond", "Low-dim H1")  # orange

                # Diagonal reference line
                max_val_candidates = []
                for arr in (high_h0, high_h1, low_h0, low_h1):
                    if arr.size:
                        max_val_candidates.append(arr[:, 1].max())
                if max_val_candidates:
                    max_val = float(np.nanmax(max_val_candidates))
                    fig.add_trace(
                        go.Scatter(
                            x=[0, max_val],
                            y=[0, max_val],
                            mode="lines",
                            line=dict(color="gray", dash="dash"),
                            showlegend=False,
                        ),
                        row=2,
                        col=2,
                    )

                fig.update_xaxes(title_text="Birth", row=2, col=2)
                fig.update_yaxes(title_text="Death", row=2, col=2)
            except Exception:
                # Fallback: annotation if plotting fails
                fig.add_annotation(
                    text="Persistence diagram\nnot available",
                    xref="x4",
                    yref="y4",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="gray"),
                )
        else:
            # Annotate unavailability of persistence diagram
            fig.add_annotation(
                text="Persistence diagram\nnot available",
                xref="x4",
                yref="y4",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="gray"),
            )

        # Update subplot axes titles
        # Only call `update_xaxes`/`update_yaxes` for 2-D plots. Attempting to
        # do so on a 3-D scene triggers Plotly validation errors, so we fall
        # back to `update_layout(scene=…)` when the original data plot is 3-D.
        if is_3d:
            # 3-D original data plot (first subplot)
            fig.update_layout(
                scene=dict(
                    xaxis_title="Dimension 1",
                    yaxis_title="Dimension 2",
                    zaxis_title="Dimension 3",
                )
            )
        else:
            fig.update_xaxes(title_text="Dimension 1", row=1, col=1)
            fig.update_yaxes(title_text="Dimension 2", row=1, col=1)

        # The UMAP embedding is always 2-D (no axis labels)

        # Update layout
        fig.update_layout(
            height=900,
            title_text="UMAP Embeddings Quality Assessment",
            title_x=0.5,
            title_font_size=18,
            showlegend=True,
            margin=dict(t=100, b=50, l=50, r=50),
            legend=dict(
                x=0.52,  # Position legend closer to persistence diagram subplot
                xanchor="left",
                y=0.15,  # Position at bottom right area (persistence diagram level)
                yanchor="bottom",
                bgcolor="rgba(255,255,255,0.8)",  # Semi-transparent background
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
                title=dict(
                    text="Persistence Diagram",
                    font=dict(size=12, color="black"),
                ),
            ),
        )

        # Store the plot HTML with unique ID for comparison mode
        if is_comparison:
            plot_config = {
                "displayModeBar": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
            }
            plot_html = pyo.plot(
                fig,
                output_type="div",
                include_plotlyjs=False,
                config=plot_config,
            )

            # Generate cuML plot as well
            embedding_cuml = comparison_data[name]["cuml"]["embedding"]
            spec_init_cuml = comparison_data[name]["cuml"]["spectral_init"]
            metrics_cuml = comparison_data[name]["cuml"]["metrics"]

            # Create a similar figure for cuML but hidden initially
            fig_cuml = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=[
                    orig_title,
                    "UMAP Embedding",
                    "Quality Metrics",
                    "Persistence Diagram",
                ],
                specs=[
                    [
                        {"type": "scatter3d" if is_3d else "scatter"},
                        {"type": "scatter"},
                    ],
                    [{"type": "table"}, {"type": "scatter"}],
                ],
                vertical_spacing=0.15,
                horizontal_spacing=0.1,
            )

            # Add original data plot (same for both implementations)
            fig_cuml.add_trace(scatter_orig, row=1, col=1)

            # Add cuML embedding
            scatter_umap_cuml = go.Scatter(
                x=embedding_cuml[:, 0],
                y=embedding_cuml[:, 1],
                mode="markers",
                marker=dict(
                    size=6,
                    color=colors,
                    colorscale="Turbo",
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="Class", side="right"),
                        x=0.48,  # Position closer to the UMAP subplot
                        xanchor="left",
                        y=0.75,  # Position at the UMAP subplot level
                        yanchor="middle",
                        len=0.4,  # Make it smaller to fit better
                        thickness=15,
                    ),
                ),
                name="UMAP Embedding",
                showlegend=False,
                hovertemplate="<b>UMAP Embedding</b><br>UMAP1: %{x:.3f}<br>UMAP2: %{y:.3f}<extra></extra>",
            )
            fig_cuml.add_trace(scatter_umap_cuml, row=1, col=2)

            # Add cuML metrics table
            metrics_data_cuml = [
                [
                    "Trustworthiness",
                    f"{metrics_cuml.get('trustworthiness', 0):.4f}",
                ],
                ["Continuity", f"{metrics_cuml.get('continuity', 0):.4f}"],
                [
                    "Geodesic Spearman Corr",
                    f"{metrics_cuml.get('geodesic_spearman_correlation', 0):.4f}",
                ],
                [
                    "Geodesic Pearson Corr",
                    f"{metrics_cuml.get('geodesic_pearson_correlation', 0):.4f}",
                ],
                ["DEMaP", f"{metrics_cuml.get('demap', 0):.4f}"],
                [
                    "Fuzzy KL (high vs low)",
                    f"{metrics_cuml.get('fuzzy_kl_divergence', 0):.4f}",
                ],
                [
                    "Fuzzy Sym KL (high vs low)",
                    f"{metrics_cuml.get('fuzzy_sym_kl_divergence', 0):.4f}",
                ],
                [
                    "Betti H0 (high→low)",
                    f"{metrics_cuml.get('betti_h0_high', 'N/A')} → {metrics_cuml.get('betti_h0_low', 'N/A')}",
                ],
                [
                    "Betti H1 (high→low)",
                    f"{metrics_cuml.get('betti_h1_high', 'N/A')} → {metrics_cuml.get('betti_h1_low', 'N/A')}",
                ],
            ]

            # Add comparison metrics if available
            if "avg_knn_recall" in metrics_cuml:
                metrics_data_cuml.append(
                    [
                        "Avg KNN Recall (vs Ref)",
                        f"{metrics_cuml['avg_knn_recall']:.4f}",
                    ]
                )
            if "mae_knn_dist" in metrics_cuml:
                metrics_data_cuml.append(
                    [
                        "KNN Distance MAE (vs Ref)",
                        f"{metrics_cuml['mae_knn_dist']:.4f}",
                    ]
                )
            if "kl_sym" in metrics_cuml:
                metrics_data_cuml.append(
                    [
                        "Fuzzy Graph Sym KL (ref vs cuML)",
                        f"{metrics_cuml['kl_sym']:.4f}",
                    ]
                )
            if "jacc" in metrics_cuml:
                metrics_data_cuml.append(
                    [
                        "Fuzzy Graph Edge Jaccard (ref vs cuML)",
                        f"{metrics_cuml['jacc']:.4f}",
                    ]
                )
            if "row_l1" in metrics_cuml:
                metrics_data_cuml.append(
                    [
                        "Fuzzy Graph Row-sum L1 (ref vs cuML)",
                        f"{metrics_cuml['row_l1']:.4f}",
                    ]
                )

            table_cuml = go.Table(
                header=dict(
                    values=["Metric", "Value"],
                    fill_color="lightblue",
                    align="left",
                    font_size=12,
                    height=30,
                ),
                cells=dict(
                    values=list(zip(*metrics_data_cuml)),
                    fill_color="white",
                    align="left",
                    font_size=11,
                    height=25,
                ),
            )
            fig_cuml.add_trace(table_cuml, row=2, col=1)

            # Add persistence diagram for cuML (reuse the same logic)
            pd_available_cuml = (
                "high_pd" in metrics_cuml and "low_pd" in metrics_cuml
            )
            if pd_available_cuml:
                try:
                    import numpy as np

                    def _extract(dgms, idx):
                        if len(dgms) > idx and len(dgms[idx]):
                            return np.asarray(dgms[idx])
                        return np.empty((0, 2))

                    high_pd_full_cuml = metrics_cuml["high_pd"]
                    low_pd_full_cuml = metrics_cuml["low_pd"]

                    high_h0_cuml = _extract(high_pd_full_cuml, 0)
                    high_h1_cuml = _extract(high_pd_full_cuml, 1)
                    low_h0_cuml = _extract(low_pd_full_cuml, 0)
                    low_h1_cuml = _extract(low_pd_full_cuml, 1)

                    high_h0_cuml = (
                        high_h0_cuml[np.isfinite(high_h0_cuml[:, 1])]
                        if high_h0_cuml.size
                        else high_h0_cuml
                    )
                    low_h0_cuml = (
                        low_h0_cuml[np.isfinite(low_h0_cuml[:, 1])]
                        if low_h0_cuml.size
                        else low_h0_cuml
                    )

                    def _plot_cuml(points, color, symbol, name):
                        if points.size:
                            fig_cuml.add_trace(
                                go.Scatter(
                                    x=points[:, 0],
                                    y=points[:, 1],
                                    mode="markers",
                                    marker=dict(
                                        size=6, color=color, symbol=symbol
                                    ),
                                    name=name,
                                ),
                                row=2,
                                col=2,
                            )

                    _plot_cuml(
                        high_h0_cuml, "#1f77b4", "circle", "High-dim H0"
                    )
                    _plot_cuml(
                        high_h1_cuml, "#2ca02c", "diamond", "High-dim H1"
                    )
                    _plot_cuml(low_h0_cuml, "#d62728", "circle", "Low-dim H0")
                    _plot_cuml(low_h1_cuml, "#ff7f0e", "diamond", "Low-dim H1")

                    max_val_candidates_cuml = []
                    for arr in (
                        high_h0_cuml,
                        high_h1_cuml,
                        low_h0_cuml,
                        low_h1_cuml,
                    ):
                        if arr.size:
                            max_val_candidates_cuml.append(arr[:, 1].max())
                    if max_val_candidates_cuml:
                        max_val_cuml = float(
                            np.nanmax(max_val_candidates_cuml)
                        )
                        fig_cuml.add_trace(
                            go.Scatter(
                                x=[0, max_val_cuml],
                                y=[0, max_val_cuml],
                                mode="lines",
                                line=dict(color="gray", dash="dash"),
                                showlegend=False,
                            ),
                            row=2,
                            col=2,
                        )

                    fig_cuml.update_xaxes(title_text="Birth", row=2, col=2)
                    fig_cuml.update_yaxes(title_text="Death", row=2, col=2)
                except Exception:
                    fig_cuml.add_annotation(
                        text="Persistence diagram\nnot available",
                        xref="x4",
                        yref="y4",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=14, color="gray"),
                    )
            else:
                fig_cuml.add_annotation(
                    text="Persistence diagram\nnot available",
                    xref="x4",
                    yref="y4",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="gray"),
                )

            # Update layout for cuML plot
            if is_3d:
                fig_cuml.update_layout(
                    scene=dict(
                        xaxis_title="Dimension 1",
                        yaxis_title="Dimension 2",
                        zaxis_title="Dimension 3",
                    )
                )
            else:
                fig_cuml.update_xaxes(title_text="Dimension 1", row=1, col=1)
                fig_cuml.update_yaxes(title_text="Dimension 2", row=1, col=1)

            # UMAP embedding axes (no labels)

            fig_cuml.update_layout(
                height=900,
                title_text="UMAP Embeddings Quality Assessment",
                title_x=0.5,
                title_font_size=18,
                showlegend=True,
                margin=dict(t=100, b=50, l=50, r=50),
                legend=dict(
                    x=0.52,  # Position legend closer to persistence diagram subplot
                    xanchor="left",
                    y=0.15,  # Position at bottom right area (persistence diagram level)
                    yanchor="bottom",
                    bgcolor="rgba(255,255,255,0.8)",  # Semi-transparent background
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1,
                    title=dict(
                        text="Persistence Diagram",
                        font=dict(size=12, color="black"),
                    ),
                ),
            )

            plot_html_cuml = pyo.plot(
                fig_cuml,
                output_type="div",
                include_plotlyjs=False,
                config=plot_config,
            )

            # Combine both plots in a container with toggle switch
            combined_html = f"""
            <div class="implementation-container" data-dataset="{name}">
                <div class="implementation-toggle-container">
                    <div class="toggle-switch">
                        <input type="checkbox" class="implementation-toggle" data-dataset="{name}">
                        <span class="toggle-slider">
                            <span class="toggle-text left">Reference</span>
                            <span class="toggle-text right">cuML</span>
                        </span>
                    </div>
                </div>
                <div class="plot-container plot-reference" id="container-{safe_name}-reference">
                    {plot_html}
                </div>
                <div class="plot-container plot-cuml" id="container-{safe_name}-cuml" style="display: none;">
                    {plot_html_cuml}
                </div>
            </div>
            """
            main_html.append(combined_html)
        else:
            # Single implementation mode
            plot_html = pyo.plot(
                fig, output_type="div", include_plotlyjs=False
            )
            main_html.append(plot_html)

        # ----------------------------------------------------------
        #   Generate spectral-initialisation scatter (2-D only)
        # ----------------------------------------------------------
        if is_comparison:
            # Handle both implementations for spectral plots
            spec_init_ref = comparison_data[name]["reference"]["spectral_init"]
            spec_init_cuml = comparison_data[name]["cuml"]["spectral_init"]

            # Generate reference spectral plot
            if spec_init_ref is not None and spec_init_ref.shape[1] >= 2:
                spec_fig_ref = go.Figure()
                spec_fig_ref.add_trace(
                    go.Scatter(
                        x=spec_init_ref[:, 0],
                        y=spec_init_ref[:, 1],
                        mode="markers",
                        marker=dict(
                            size=6,
                            color=colors,
                            colorscale="Turbo",
                            showscale=False,
                        ),
                        showlegend=False,
                        hovertemplate="<b>Spectral Init (Reference)</b><br>S1: %{x:.3f}<br>S2: %{y:.3f}<extra></extra>",
                    )
                )
                spec_fig_ref.update_xaxes(title_text="Spectral-Dim 1")
                spec_fig_ref.update_yaxes(title_text="Spectral-Dim 2")
                spec_fig_ref.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    title=f"Spectral Init – {name} (Reference)",
                )
                spec_plot_html_ref = pyo.plot(
                    spec_fig_ref, output_type="div", include_plotlyjs=False
                )
            else:
                spec_plot_html_ref = (
                    "<p style='color:gray;'>Spectral init not available</p>"
                )

            # Generate cuML spectral plot
            if spec_init_cuml is not None and spec_init_cuml.shape[1] >= 2:
                spec_fig_cuml = go.Figure()
                spec_fig_cuml.add_trace(
                    go.Scatter(
                        x=spec_init_cuml[:, 0],
                        y=spec_init_cuml[:, 1],
                        mode="markers",
                        marker=dict(
                            size=6,
                            color=colors,
                            colorscale="Turbo",
                            showscale=False,
                        ),
                        showlegend=False,
                        hovertemplate="<b>Spectral Init (cuML)</b><br>S1: %{x:.3f}<br>S2: %{y:.3f}<extra></extra>",
                    )
                )
                spec_fig_cuml.update_xaxes(title_text="Spectral-Dim 1")
                spec_fig_cuml.update_yaxes(title_text="Spectral-Dim 2")
                spec_fig_cuml.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    title=f"Spectral Init – {name} (cuML)",
                )
                spec_plot_html_cuml = pyo.plot(
                    spec_fig_cuml, output_type="div", include_plotlyjs=False
                )
            else:
                spec_plot_html_cuml = (
                    "<p style='color:gray;'>Spectral init not available</p>"
                )

            # Combine spectral plots (controlled by main toggle)
            combined_spectral_html = f"""
            <div class="spectral-comparison-container">
                <div class="spectral-plot-container spectral-reference" id="spectral-container-{safe_name}-reference">
                    {spec_plot_html_ref}
                </div>
                <div class="spectral-plot-container spectral-cuml" id="spectral-container-{safe_name}-cuml" style="display: none;">
                    {spec_plot_html_cuml}
                </div>
            </div>
            """
            spectral_html.append(combined_spectral_html)
        else:
            # Single implementation mode
            if spec_init is not None and spec_init.shape[1] >= 2:
                spec_fig = go.Figure()
                spec_fig.add_trace(
                    go.Scatter(
                        x=spec_init[:, 0],
                        y=spec_init[:, 1],
                        mode="markers",
                        marker=dict(
                            size=6,
                            color=colors,
                            colorscale="Turbo",
                            showscale=False,
                        ),
                        showlegend=False,
                        hovertemplate="<b>Spectral Init</b><br>S1: %{x:.3f}<br>S2: %{y:.3f}<extra></extra>",
                    )
                )

                spec_fig.update_xaxes(title_text="Spectral-Dim 1")
                spec_fig.update_yaxes(title_text="Spectral-Dim 2")
                spec_fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    title=f"Spectral Init – {name}",
                )
                spec_plot_html = pyo.plot(
                    spec_fig, output_type="div", include_plotlyjs=False
                )
            else:
                spec_plot_html = (
                    "<p style='color:gray;'>Spectral init not available</p>"
                )

            spectral_html.append(spec_plot_html)

    return main_html, spectral_html, comparison_data


def generate_web_report(datasets, embeddings, all_metrics, spectral_inits):
    """Generate complete HTML web page with all results, including spectral initialisation.

    The spectral-initialisation scatter plot for each dataset is hidden inside a
    hover-triggered bubble that keeps the overall layout unchanged aside from a
    small button labelled "Spectral&nbsp;Init".
    """

    # Create plots (main + spectral init)
    plot_content, spectral_plots, comparison_data = create_plotly_plots(
        datasets, embeddings, all_metrics, spectral_inits
    )

    # Create HTML structure
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>UMAP Quality Assessment Results</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/3.0.3/plotly.min.js"></script>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f8f9fa;
            }}
            .header {{
                text-align: center;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .dataset-section {{
                background: white;
                margin: 30px auto;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                max-width: 1200px;
            }}
            .metrics-summary {{
                background: #e9ecef;
                padding: 20px;
                border-radius: 8px;
                margin: 20px auto;
                max-width: 1200px;
            }}
            .summary-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            .summary-table th, .summary-table td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            .summary-table th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            /* Spectral-init hover bubble */
            .spectral-bubble-container {{
                position: relative;
                display: inline-block;
                margin-bottom: 10px;
            }}
            .spectral-btn {{
                background-color: #667eea;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                cursor: pointer;
                font-size: 0.9rem;
            }}
            .spectral-bubble {{
                display: none;
                position: absolute;
                z-index: 99;
                left: 50%;
                transform: translateX(-50%);
                top: 110%;
                background: white;
                border: 1px solid #ddd;
                box-shadow: 0 2px 6px rgba(0,0,0,0.15);
                border-radius: 8px;
                padding: 10px;
                width: 450px;
            }}
            .spectral-bubble-container:hover .spectral-bubble {{
                display: block;
            }}

            /* Implementation toggle switch styles */
            .implementation-toggle-container {{
                position: absolute;
                top: 20px;
                right: 20px;
                z-index: 100;
            }}
            .toggle-switch {{
                position: relative;
                display: inline-block;
                width: 160px;
                height: 40px;
                cursor: pointer;
            }}
            .toggle-switch input {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                opacity: 0;
                cursor: pointer;
                z-index: 3;
                margin: 0;
            }}
            .toggle-slider {{
                position: absolute;
                cursor: pointer;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: #667eea;
                border-radius: 20px;
                transition: 0.3s;
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 0 15px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.15);
                border: 2px solid white;
            }}
            .toggle-slider:before {{
                position: absolute;
                content: "";
                height: 32px;
                width: 72px;
                left: 4px;
                bottom: 4px;
                background-color: white;
                border-radius: 16px;
                transition: 0.3s;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }}
            .toggle-switch input:checked + .toggle-slider {{
                background-color: #28a745;
            }}
            .toggle-switch input:checked + .toggle-slider:before {{
                transform: translateX(80px);
            }}
            .toggle-text {{
                font-size: 0.85rem;
                font-weight: bold;
                color: white;
                z-index: 1;
                position: relative;
            }}
            .toggle-text.left {{
                margin-left: 0;
            }}
            .toggle-text.right {{
                margin-right: 0;
            }}
            .plot-container {{
                transition: opacity 0.3s ease-in-out;
            }}
            .spectral-plot-container {{
                transition: opacity 0.3s ease-in-out;
            }}
            .spectral-comparison-container {{
                margin-top: 10px;
            }}
            .implementation-container {{
                margin-bottom: 30px;
                position: relative;
                padding-top: 70px;
            }}

            h1 {{
                margin: 0;
                font-size: 2.5rem;
            }}
            h2 {{
                color: #495057;
                border-bottom: 2px solid #667eea;
                padding-bottom: 10px;
            }}
            .subtitle {{
                font-size: 1.2rem;
                margin-top: 10px;
                opacity: 0.9;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>UMAP Quality Assessment Results</h1>
            <div class="subtitle">
                Comprehensive analysis of UMAP embeddings across synthetic and classic datasets<br>
                                 Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>

        <div class="metrics-summary">
            <h2>Summary of All Datasets</h2>
            <table class="summary-table">
                <thead>
                    <tr>
                        <th>Dataset</th>
                        <th>Input Shape</th>
                        <th>Trustworthiness</th>
                        <th>Continuity</th>
                        <th>Geo Spearman</th>
                        <th>Geo Pearson</th>
                        <th>DEMaP</th>
                        <th>Fuzzy KL</th>
                        <th>Fuzzy Sym KL</th>
                        <th>Avg KNN Recall</th>
                        <th>KNN Dist MAE</th>
                        <th>Fuzzy Graph Sym KL (ref vs cuML)</th>
                        <th>Edge Jaccard (ref vs cuML)</th>
                        <th>Row-sum L1 (ref vs cuML)</th>
                    </tr>
                </thead>
                <tbody>
    """

    # Add summary table rows
    for name, (X, _) in datasets.items():
        # Check if this is comparison mode
        if name in comparison_data:
            # Add rows for both implementations
            for impl_name, impl_data in [
                ("Reference", comparison_data[name]["reference"]),
                ("cuML", comparison_data[name]["cuml"]),
            ]:
                metrics = impl_data["metrics"]

                # Extract comparison metrics if available and format safely
                avg_knn_recall = metrics.get("avg_knn_recall")
                avg_knn_recall_str = (
                    f"{avg_knn_recall:.4f}"
                    if isinstance(avg_knn_recall, (int, float))
                    else "N/A"
                )
                mae_knn_dist = metrics.get("mae_knn_dist")
                mae_knn_dist_str = (
                    f"{mae_knn_dist:.4f}"
                    if isinstance(mae_knn_dist, (int, float))
                    else "N/A"
                )
                kl_sym_val = metrics.get("kl_sym")
                kl_sym_str = (
                    f"{kl_sym_val:.4f}"
                    if isinstance(kl_sym_val, (int, float))
                    else "N/A"
                )
                jacc_val = metrics.get("jacc")
                jacc_str = (
                    f"{jacc_val:.4f}"
                    if isinstance(jacc_val, (int, float))
                    else "N/A"
                )
                row_l1_val = metrics.get("row_l1")
                row_l1_str = (
                    f"{row_l1_val:.4f}"
                    if isinstance(row_l1_val, (int, float))
                    else "N/A"
                )

                html += f"""
                            <tr>
                                <td><strong>{name} ({impl_name})</strong></td>
                                <td>{X.shape}</td>
                                <td>{metrics.get('trustworthiness', 0):.4f}</td>
                                <td>{metrics.get('continuity', 0):.4f}</td>
                                <td>{metrics.get('geodesic_spearman_correlation', 0):.4f}</td>
                                <td>{metrics.get('geodesic_pearson_correlation', 0):.4f}</td>
                                <td>{metrics.get('demap', 0):.4f}</td>
                                <td>{metrics.get('fuzzy_kl_divergence', 0):.4f}</td>
                                <td>{metrics.get('fuzzy_sym_kl_divergence', 0):.4f}</td>
                                <td>{avg_knn_recall_str}</td>
                                <td>{mae_knn_dist_str}</td>
                                <td>{kl_sym_str}</td>
                                <td>{jacc_str}</td>
                                <td>{row_l1_str}</td>
                            </tr>
                """
        else:
            # Single implementation mode
            metrics = all_metrics[name]

            # Extract comparison metrics if available and format safely
            avg_knn_recall = metrics.get("avg_knn_recall")
            avg_knn_recall_str = (
                f"{avg_knn_recall:.4f}"
                if isinstance(avg_knn_recall, (int, float))
                else "N/A"
            )
            mae_knn_dist = metrics.get("mae_knn_dist")
            mae_knn_dist_str = (
                f"{mae_knn_dist:.4f}"
                if isinstance(mae_knn_dist, (int, float))
                else "N/A"
            )
            kl_sym_val = metrics.get("kl_sym")
            kl_sym_str = (
                f"{kl_sym_val:.4f}"
                if isinstance(kl_sym_val, (int, float))
                else "N/A"
            )
            jacc_val = metrics.get("jacc")
            jacc_str = (
                f"{jacc_val:.4f}"
                if isinstance(jacc_val, (int, float))
                else "N/A"
            )
            row_l1_val = metrics.get("row_l1")
            row_l1_str = (
                f"{row_l1_val:.4f}"
                if isinstance(row_l1_val, (int, float))
                else "N/A"
            )

            html += f"""
                        <tr>
                            <td><strong>{name}</strong></td>
                            <td>{X.shape}</td>
                            <td>{metrics.get('trustworthiness', 0):.4f}</td>
                            <td>{metrics.get('continuity', 0):.4f}</td>
                            <td>{metrics.get('geodesic_spearman_correlation', 0):.4f}</td>
                            <td>{metrics.get('geodesic_pearson_correlation', 0):.4f}</td>
                            <td>{metrics.get('demap', 0):.4f}</td>
                            <td>{metrics.get('fuzzy_kl_divergence', 0):.4f}</td>
                            <td>{metrics.get('fuzzy_sym_kl_divergence', 0):.4f}</td>
                            <td>{avg_knn_recall_str}</td>
                            <td>{mae_knn_dist_str}</td>
                            <td>{kl_sym_str}</td>
                            <td>{jacc_str}</td>
                            <td>{row_l1_str}</td>
                        </tr>
            """

    html += """
                </tbody>
            </table>
        </div>
    """

    # Add individual dataset sections
    for i, (name, _) in enumerate(datasets.items()):
        plot_html = plot_content[i]
        spectral_html = spectral_plots[i]
        html += f"""
        <div class="dataset-section">
            <h2>{name}</h2>
            <div class="spectral-bubble-container">
                <button class="spectral-btn">Spectral Init</button>
                <div class="spectral-bubble">{spectral_html}</div>
            </div>
            {plot_html}
        </div>
        """

    html += """
        <div style="text-align: center; margin-top: 50px; padding: 20px; color: #6c757d;">
            <p>This analysis evaluates embedding quality using multiple metrics including trustworthiness,
               continuity, geodesic correlation, fuzzy simplicial set cross-entropy, and topological features.</p>
        </div>

        <script>
            // Implementation toggle functionality
            document.addEventListener('DOMContentLoaded', function() {{
                console.log('Toggle script loaded');
                const toggles = document.querySelectorAll('.implementation-toggle');
                console.log('Found toggles:', toggles.length);

                toggles.forEach(function(toggle, index) {{
                    console.log('Setting up toggle ' + index + ' for dataset:', toggle.getAttribute('data-dataset'));

                    toggle.addEventListener('change', function(e) {{
                        console.log('Toggle changed:', this.checked);
                        const dataset = this.getAttribute('data-dataset');
                        const isChecked = this.checked;
                        // Use regex to replace all spaces and special characters for safe ID
                        const safeName = dataset.replace(/[^a-zA-Z0-9]/g, '-').toLowerCase();

                        console.log('Dataset:', dataset, 'SafeName:', safeName, 'Checked:', isChecked);

                        // Main plot containers
                        const refContainer = document.getElementById('container-' + safeName + '-reference');
                        const cumlContainer = document.getElementById('container-' + safeName + '-cuml');

                        // Spectral plot containers
                        const refSpectralContainer = document.getElementById('spectral-container-' + safeName + '-reference');
                        const cumlSpectralContainer = document.getElementById('spectral-container-' + safeName + '-cuml');

                        console.log('Found containers: ref=' + !!refContainer + ', cuml=' + !!cumlContainer + ', refSpectral=' + !!refSpectralContainer + ', cumlSpectral=' + !!cumlSpectralContainer);

                        if (!isChecked) {{
                            // Show reference implementation (both main and spectral)
                            console.log('Switching to Reference');
                            if (refContainer) {{
                                refContainer.style.display = 'block';
                                refContainer.style.opacity = '1';
                                // Resize Plotly plots after showing
                                setTimeout(function() {{
                                    const plotDivs = refContainer.querySelectorAll('.plotly-graph-div');
                                    plotDivs.forEach(function(plotDiv) {{
                                        if (window.Plotly && window.Plotly.Plots) {{
                                            window.Plotly.Plots.resize(plotDiv);
                                        }}
                                    }});
                                }}, 100);
                            }}
                            if (cumlContainer) {{
                                cumlContainer.style.display = 'none';
                                cumlContainer.style.opacity = '0';
                            }}
                            if (refSpectralContainer) {{
                                refSpectralContainer.style.display = 'block';
                                refSpectralContainer.style.opacity = '1';
                                // Resize spectral plots
                                setTimeout(function() {{
                                    const plotDivs = refSpectralContainer.querySelectorAll('.plotly-graph-div');
                                    plotDivs.forEach(function(plotDiv) {{
                                        if (window.Plotly && window.Plotly.Plots) {{
                                            window.Plotly.Plots.resize(plotDiv);
                                        }}
                                    }});
                                }}, 100);
                            }}
                            if (cumlSpectralContainer) {{
                                cumlSpectralContainer.style.display = 'none';
                                cumlSpectralContainer.style.opacity = '0';
                            }}
                        }} else {{
                            // Show cuML implementation (both main and spectral)
                            console.log('Switching to cuML');
                            if (refContainer) {{
                                refContainer.style.display = 'none';
                                refContainer.style.opacity = '0';
                            }}
                            if (cumlContainer) {{
                                cumlContainer.style.display = 'block';
                                cumlContainer.style.opacity = '1';
                                // Resize Plotly plots after showing
                                setTimeout(function() {{
                                    const plotDivs = cumlContainer.querySelectorAll('.plotly-graph-div');
                                    plotDivs.forEach(function(plotDiv) {{
                                        if (window.Plotly && window.Plotly.Plots) {{
                                            window.Plotly.Plots.resize(plotDiv);
                                        }}
                                    }});
                                }}, 100);
                            }}
                            if (refSpectralContainer) {{
                                refSpectralContainer.style.display = 'none';
                                refSpectralContainer.style.opacity = '0';
                            }}
                            if (cumlSpectralContainer) {{
                                cumlSpectralContainer.style.display = 'block';
                                cumlSpectralContainer.style.opacity = '1';
                                // Resize spectral plots
                                setTimeout(function() {{
                                    const plotDivs = cumlSpectralContainer.querySelectorAll('.plotly-graph-div');
                                    plotDivs.forEach(function(plotDiv) {{
                                        if (window.Plotly && window.Plotly.Plots) {{
                                            window.Plotly.Plots.resize(plotDiv);
                                        }}
                                    }});
                                }}, 100);
                            }}
                        }}
                    }});
                }});
            }});
        </script>
    </body>
    </html>
    """

    return html
