"""
Visualization utilities for hyperbolic embeddings in the Poincaré ball model.

This module provides functions to visualize embeddings in the Poincaré ball model,
including 2D and 3D visualizations of hyperbolic space.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def visualize_poincare_ball_2d(points, labels=None, colors=None, ax=None, show_boundary=True, 
                               title="Poincaré Ball Embeddings (2D Projection)", figsize=(10, 10),
                               alpha=0.7, s=100, annotate=False, cmap='viridis'):
    """
    Visualize points in a 2D Poincaré ball using matplotlib.
    
    Args:
        points: Array of points, shape (n_points, 2) or (n_points, >2) for projection
        labels: Optional labels for the points
        colors: Optional colors for the points
        ax: Optional matplotlib axis to plot on
        show_boundary: Whether to show the unit circle boundary
        title: Plot title
        figsize: Figure size
        alpha: Transparency of points
        s: Size of points
        annotate: Whether to annotate points with labels
        cmap: Colormap for points if colors are not provided
        
    Returns:
        matplotlib axis with the plot
    """
    # If points have more than 2 dimensions, take first 2
    if points.shape[1] > 2:
        points = points[:, :2]
    
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the boundary of the Poincaré ball (unit circle)
    if show_boundary:
        circle = Circle((0, 0), 1, fill=False, color='black', linestyle='--', alpha=0.5)
        ax.add_patch(circle)
    
    # Plot points
    scatter = ax.scatter(points[:, 0], points[:, 1], c=colors, cmap=cmap, alpha=alpha, s=s)
    
    # Add annotations if requested
    if annotate and labels is not None:
        for i, (p, l) in enumerate(zip(points, labels)):
            ax.annotate(l, (p[0], p[1]), fontsize=8)
    
    # Add colorbar if colors were numeric
    if colors is not None and not isinstance(colors[0], str):
        plt.colorbar(scatter, ax=ax)
    
    # Set plot properties
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    
    return ax

def visualize_poincare_ball_3d(points, labels=None, colors=None, 
                               title="Poincaré Ball Embeddings (3D Projection)",
                               show_boundary=True, opacity=0.7, size=6, annotate=False):
    """
    Visualize points in a 3D Poincaré ball using Plotly.
    
    Args:
        points: Array of points, shape (n_points, 3) or (n_points, >3) for projection
        labels: Optional labels for the points
        colors: Optional colors for the points
        title: Plot title
        show_boundary: Whether to show the unit sphere boundary
        opacity: Transparency of points
        size: Size of points
        annotate: Whether to annotate points with labels
        
    Returns:
        Plotly figure with the 3D plot
    """
    # If points have more than 3 dimensions, take first 3
    if points.shape[1] > 3:
        points = points[:, :3]
    
    # Create figure
    fig = go.Figure()
    
    # Add points
    fig.add_trace(go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers' + ('+text' if annotate else ''),
        marker=dict(
            size=size,
            color=colors,
            opacity=opacity
        ),
        text=labels,
        textposition='top center'
    ))
    
    # Add unit sphere boundary if requested
    if show_boundary:
        # Create a mesh grid for a unit sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Add the unit sphere as a surface with high transparency
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            opacity=0.1,
            showscale=False,
            colorscale=[[0, 'rgb(240,240,240)'], [1, 'rgb(200,200,200)']]
        ))
    
    # Set layout properties
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=[-1.1, 1.1], autorange=False),
            yaxis=dict(range=[-1.1, 1.1], autorange=False),
            zaxis=dict(range=[-1.1, 1.1], autorange=False),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        template="plotly_white"
    )
    
    return fig

def plot_geodesic(start_point, end_point, manifold, num_points=50, ax=None, color='red', linewidth=2, alpha=0.7):
    """
    Plot a geodesic between two points in the Poincaré ball.
    
    Args:
        start_point: Starting point in the Poincaré ball
        end_point: Ending point in the Poincaré ball
        manifold: Poincaré ball manifold object with geodesic function
        num_points: Number of points to sample along the geodesic
        ax: Matplotlib axis to plot on
        color: Line color
        linewidth: Line width
        alpha: Line transparency
        
    Returns:
        matplotlib axis with the geodesic plotted
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Sample points along the geodesic
    t_values = np.linspace(0, 1, num_points)
    geodesic_points = np.stack([manifold.geodesic(start_point, end_point, t).numpy() for t in t_values])
    
    # Plot the geodesic
    ax.plot(geodesic_points[:, 0], geodesic_points[:, 1], color=color, linewidth=linewidth, alpha=alpha)
    
    return ax

def plot_multiple_geodesics(points, manifold, ax=None, colors=None, linewidth=2, alpha=0.7):
    """
    Plot geodesics between multiple pairs of points in the Poincaré ball.
    
    Args:
        points: Array of point pairs, shape (n_pairs, 2, dim)
        manifold: Poincaré ball manifold object with geodesic function
        ax: Matplotlib axis to plot on
        colors: Colors for each geodesic
        linewidth: Line width
        alpha: Line transparency
        
    Returns:
        matplotlib axis with the geodesics plotted
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(points)))
    
    for i, (start, end) in enumerate(points):
        plot_geodesic(start, end, manifold, ax=ax, color=colors[i], linewidth=linewidth, alpha=alpha)
    
    return ax

def create_embeddings_visualization(embeddings, labels=None, title="Hyperbolic Embeddings",
                                   manifold=None, geodesics=None, save_path=None):
    """
    Create a comprehensive visualization of embeddings in hyperbolic space.
    
    Args:
        embeddings: Array of embeddings
        labels: Optional labels for embeddings
        title: Plot title
        manifold: Optional manifold object for geodesic calculations
        geodesics: Optional list of point pairs for geodesic visualization
        save_path: Path to save the visualization to
        
    Returns:
        Figure objects for both 2D and 3D visualizations
    """
    # For 2D visualization
    fig_2d, ax = plt.subplots(figsize=(12, 12))
    visualize_poincare_ball_2d(embeddings, labels=labels, ax=ax, title=f"{title} (2D)")
    
    # Plot geodesics if provided
    if geodesics is not None and manifold is not None:
        plot_multiple_geodesics(geodesics, manifold, ax=ax)
    
    # For 3D visualization if we have at least 3 dimensions
    if embeddings.shape[1] >= 3:
        fig_3d = visualize_poincare_ball_3d(embeddings, labels=labels, title=f"{title} (3D)")
    else:
        fig_3d = None
    
    # Save if requested
    if save_path:
        fig_2d.savefig(f"{save_path}_2d.png", dpi=300, bbox_inches='tight')
        if fig_3d:
            fig_3d.write_html(f"{save_path}_3d.html")
    
    return fig_2d, fig_3d