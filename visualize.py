import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def visualize_tensor(tensor, x, y, width=20, height=8, pic_settings=None, vmin=-25, vmax=25):
    
    row = np.shape(tensor)[0]
    col = np.shape(tensor)[1] 
    
    fig, axes = plt.subplots(row, col, figsize=(width, height))
    # fig.set_size_inches(25, 25)
    fig.tight_layout()

    if pic_settings is None: 
        pic_settings = {
            'cmap' : 'RdBu_r'
        }
        
    # Set color map with white in the middle
    if 'cmap' not in pic_settings:
        pic_settings['cmap'] = 'RdBu_r'
                
    pic_settings['vmin'] = vmin
    pic_settings['vmax'] = vmax
    
    def common_settings(ax):
        ax.set_aspect('equal')
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        
    for i in range(0, row):
        for j in range(0, col):
            if row == 1 and col == 1:
                a = axes.pcolormesh(x, y, tensor[i][j], **pic_settings)
                common_settings(axes)
            elif row == 1 and col > 1:
                a = axes[j].pcolormesh(x, y, tensor[i][j], **pic_settings)
                common_settings(axes[j])
            elif col == 1 and row > 1:
                a = axes[i].pcolormesh(x, y, tensor[i][j], **pic_settings)
                # common_settings(axes[j])
            else:
                a = axes[i][j].pcolormesh(x, y, tensor[i][j], **pic_settings)
                common_settings(axes[i][j])
    
    return fig, axes

    # clb = fig.colorbar(a, ax=axes)
    # clb.ax.set_ylabel('Units', labelpad=50, rotation=0)