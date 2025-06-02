import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from laguerre_model import make_step_response_function

def create_interactive_step_response_plot():
    """
    Creates an interactive step response plot with sliders for parameters p and c vector.
    """
    
    # Fixed parameters
    N = 4
    nT = 20
    T = 0.5
    t = T * np.arange(nT+1)
    
    # Create the step response function
    step_response = make_step_response_function(T, nT)
    
    # Create figure and subplots
    fig = plt.figure(figsize=(10, 8))
    
    # Main plot area - leave space at bottom for sliders
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.1, 
                         left=0.1, right=0.95, top=0.85, bottom=0.35)
    
    ax1 = fig.add_subplot(gs[0])  # States plot
    ax2 = fig.add_subplot(gs[1])  # Output plot
    
    # Initial parameter values
    p_init = 0.5
    c_init = np.ones(N)
    
    # Compute initial step response
    X_init, Y_init = step_response(p_init, c_init)
    X_init = np.array(X_init)
    Y_init = np.array(Y_init)
    
    # Create initial plots
    x_labels = [f"$x_{i+1}(t)$" for i in range(N)]
    y_label = "$y(t)$"
    
    # Plot states
    lines_x = []
    for i in range(N):
        line, = ax1.plot(t, X_init[:, i], marker='.', linestyle='-', label=x_labels[i])
        lines_x.append(line)
    
    ax1.grid(True)
    ax1.legend()
    ax1.set_ylabel('States')
    ax1.set_title('Laguerre Model Step Response')
    
    # Plot output
    line_y, = ax2.plot(t, Y_init, marker='.', linestyle='-', label=y_label, color='red')
    ax2.grid(True)
    ax2.legend()
    ax2.set_xlabel('Time (t)')
    ax2.set_ylabel('Output')
    
    # Create slider axes
    slider_height = 0.03
    slider_spacing = 0.04
    slider_left = 0.15
    slider_width = 0.7
    
    # Parameter p slider
    ax_p = fig.add_axes([slider_left, 0.25, slider_width, slider_height])
    slider_p = Slider(ax_p, 'p', 0.1, 0.9, valinit=p_init, valfmt='%.2f')
    
    # Parameter c sliders
    sliders_c = []
    for i in range(N):
        ax_c = fig.add_axes([slider_left, 0.25 - (i+1)*slider_spacing, slider_width, slider_height])
        slider_c = Slider(ax_c, f'c[{i+1}]', 0.1, 2.0, valinit=c_init[i], valfmt='%.2f')
        sliders_c.append(slider_c)
    
    def update_plot(val=None):
        """Update the plot when sliders change"""
        # Get current parameter values
        p_current = slider_p.val
        c_current = np.array([slider.val for slider in sliders_c])
        
        try:
            # Compute new step response
            X_new, Y_new = step_response(p_current, c_current)
            X_new = np.array(X_new)
            Y_new = np.array(Y_new)
            
            # Update state plots
            for i, line in enumerate(lines_x):
                line.set_ydata(X_new[:, i])
            
            # Update output plot
            line_y.set_ydata(Y_new)
            
            # Rescale axes if needed
            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()
            
            # Redraw
            fig.canvas.draw_idle()
            
        except Exception as e:
            print(f"Error updating plot: {e}")
    
    # Connect sliders to update function
    slider_p.on_changed(update_plot)
    for slider in sliders_c:
        slider.on_changed(update_plot)
    
    # Instructions
    fig.suptitle('Interactive Step Response Plot\nAdjust sliders to change parameters p and c', 
                 fontsize=12, y=0.95)
    
    plt.show()
    
    return fig, (ax1, ax2), slider_p, sliders_c

# Usage instructions for Jupyter notebook:
print("To use this interactive plot in Jupyter notebook:")
print("1. Make sure you have: %matplotlib widget")
print("2. Run: fig, axes, slider_p, sliders_c = create_interactive_step_response_plot()")
print("3. Adjust the sliders to see real-time updates!")

# Create the interactive plot
if __name__ == "__main__":
    # For standalone execution
    fig, axes, slider_p, sliders_c = create_interactive_step_response_plot()