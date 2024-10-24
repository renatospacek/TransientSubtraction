#%%
import h5py
import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import matplotlib.ticker as ticker

plt.rcParams['axes.grid'] = False
plt.rcParams['text.usetex'] = True
plt.rcParams["xtick.direction"] = plt.rcParams["ytick.direction"] = "in"

def extract_values(f):
    M = 10**5;
    sem1 = np.sqrt(f['v1'][:]/M);
    sem2 = np.sqrt(f['v2'][:]/M);
    semd = np.sqrt(f['vd'][:]/M);
    sem1c = np.sqrt(f['cv1'][:]/M);
    sem2c = np.sqrt(f['cv2'][:]/M);
    semdc = np.sqrt(f['cvd'][:]/M);

    return (
        f['R1'][:],
        f['R2'][:],
        f['dR'][:],
        f['cR1'][:],
        f['cR2'][:],
        f['cdR'][:],
        f['v1'][:],
        f['v2'][:],
        f['vd'][:],
        f['cv1'][:],
        f['cv2'][:],
        f['cvd'][:],
        f['tv'][:],
        f['ηv'][:],
        M,
        sem1,
        sem2,
        semd,
        sem1c,
        sem2c,
        semdc,
    )

def plot_with_fill(ax, x, y, y_sem, alpha, color):
    line, = ax.plot(x, y)
    ax.fill_between(x, y + y_sem, y - y_sem, alpha=alpha, facecolor=color)
    return line

# %% ==========================================================================
# ########## LJ mobility ##########
#==============================================================================
f = h5py.File('upt_mobility_data.jld2', 'r')
R1, R2, dR, cR1, cR2, cdR, v1, v2, vd, cv1, cv2, cvd, tv, eta, M, sem1, sem2, semd, sem1c, sem2c, semdc = extract_values(f)

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
tc_ref = 0.122; ind = 0;

fig.suptitle(r"$\eta = {}$".format(eta[ind]), fontsize=22)

# Intantaneous response subplot
l1 = plot_with_fill(ax1, tv, R1[:, ind]/eta[ind], sem1[:, ind]/eta[ind], 0, 'C0')
l2 = plot_with_fill(ax1, tv, R2[:, ind]/eta[ind], sem2[:, ind]/eta[ind], 0.25, 'C1')
l3 = plot_with_fill(ax1, tv, dR[:, ind]/eta[ind], semd[:, ind]/eta[ind], 0.25, 'C2')
ax1.set_xlim(0, 2)

# Integrated response subplot
plot_with_fill(ax2, tv, cR1[:, ind], sem1c[:, ind], 0, 'C0')
plot_with_fill(ax2, tv, cR2[:, ind], sem2c[:, ind], 0.25, 'C1')
plot_with_fill(ax2, tv, cdR[:, ind], semdc[:, ind], 0.25, 'C2')
ax2.axhline(y=tc_ref, color='k', linestyle='--')

# Zoom on the second subplot
if ind == 0:
    axins = inset_axes(ax2, width="50%", height="30%", loc='lower left')
    axins.set_ylim(0.105, 0.135)
elif ind == 1:
    axins = inset_axes(ax2, width="100%", height="100%", bbox_to_anchor=(0.1, 0.2, 0.5, 0.4), bbox_transform=ax2.transAxes, loc='upper left')
    axins.set_ylim(0.115, 0.1375)
elif ind == 2:
    axins = inset_axes(ax2, width="100%", height="100%", bbox_to_anchor=(0.1, 0.2, 0.5, 0.4), bbox_transform=ax2.transAxes, loc='upper left')
    axins.set_ylim(0.1175, 0.1275)

plot_with_fill(axins, tv, cR1[:, ind], sem1c[:, ind], 0, 'C0')
plot_with_fill(axins, tv, cR2[:, ind], sem2c[:, ind], 0.25, 'C1')
plot_with_fill(axins, tv, cdR[:, ind], semdc[:, ind], 0.25, 'C2')

axins.axhline(y=tc_ref, color='k', linestyle='--')
axins.set_xlim(ax2.get_xlim())
axins.set_xticklabels([])
axins.yaxis.set_ticks_position('right')
axins.yaxis.set_tick_params(labelsize=14)
ax2.indicate_inset_zoom(axins, lw=0)

# Labels and legends
ax1.set_ylabel(r'Response $R/\eta$', fontsize=20)
ax2.set_xlabel('Time', fontsize=20)
ax2.set_ylabel(r'Estimator $\widehat{\rho}$', fontsize=20)
ax1.legend([l1, l2, l3], ['Equil.', 'Trans.', 'Subtr.'], fontsize=20, loc='best', ncol=3, columnspacing=0.8, handlelength=1)

# Axis tick configurations
ax2.xaxis.set_major_locator(MultipleLocator(0.5))
ax2.xaxis.set_minor_locator(MultipleLocator(0.25))
ax1.yaxis.set_tick_params(labelsize=16)
ax2.xaxis.set_tick_params(labelsize=16)
ax2.yaxis.set_tick_params(labelsize=16)

yaxis_locators = {
    0: (MultipleLocator(1), MultipleLocator(0.3)),
    1: (MultipleLocator(0.5), MultipleLocator(0.05))
}

if ind in yaxis_locators:
    ax1.yaxis.set_major_locator(yaxis_locators[ind][0])
    ax2.yaxis.set_major_locator(yaxis_locators[ind][1])

fig.align_ylabels()
plt.show()
# fig.savefig('LJ_mobility_{}.pdf'.format(eta[ind]))

# %% ==========================================================================
# ########## LJ shear ##########
#==============================================================================
yratio = "1.0"; zratio = "1.0"
# yratio = "2.0"; zratio = "0.5"
f = h5py.File(f'upt_sv_data_{yratio}_{zratio}.jld2', 'r')
R1, R2, dR, cR1, cR2, cdR, v1, v2, vd, cv1, cv2, cvd, tv, eta, M, sem1, sem2, semd, sem1c, sem2c, semdc = extract_values(f);

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
tc_ref = 0.322; ind = 1;

fig.suptitle(r"$\eta = {}$".format(eta[ind]), fontsize = '22')

# Intantaneous response subplot
l1 = plot_with_fill(ax1, tv, R1[:, ind]/eta[ind], sem1[:, ind]/eta[ind], 0, 'C0')
l2 = plot_with_fill(ax1, tv, R2[:, ind]/eta[ind], sem2[:, ind]/eta[ind], 0.25, 'C1')
l3 = plot_with_fill(ax1, tv, dR[:, ind]/eta[ind], semd[:, ind]/eta[ind], 0.25, 'C2')
ax1.set_xlim(0, 3.5)

# Integrated response subplot
plot_with_fill(ax2, tv, cR1[:, ind], sem1c[:, ind], 0, 'C0')
plot_with_fill(ax2, tv, cR2[:, ind], sem2c[:, ind], 0.25, 'C1')
plot_with_fill(ax2, tv, cdR[:, ind], semdc[:, ind], 0.25, 'C2')
ax2.axhline(y=tc_ref, color='k', linestyle='--')

# Zoom on the second subplot
if ind == 0:
    axins = inset_axes(ax2, width="100%", height="100%", bbox_to_anchor=(0.35, 0.3, 0.5, 0.4), bbox_transform=ax2.transAxes)#, loc='upper left')
    axins.set_xlim(2, 3.5)
    axins.set_ylim(0.3025, 0.3325)
    axins.yaxis.set_major_locator(ticker.FixedLocator([0.31, 0.32, 0.33]))

elif ind == 1:
    axins = inset_axes(ax2, width="100%", height="100%", bbox_to_anchor=(0.35, 0.3, 0.5, 0.4), bbox_transform=ax2.transAxes)#, loc='upper left')
    axins.set_xlim(2.5, 3.5)
    axins.xaxis.set_major_locator(MultipleLocator(0.5))
    axins.xaxis.set_minor_locator(MultipleLocator(0.25))
    axins.set_ylim(0.316, 0.324)
    axins.yaxis.set_major_locator(ticker.FixedLocator([0.317, 0.32, 0.323]))  

plot_with_fill(axins, tv, cR1[:, ind], sem1c[:, ind], 0, 'C0')
plot_with_fill(axins, tv, cR2[:, ind], sem2c[:, ind], 0.25, 'C1')
plot_with_fill(axins, tv, cdR[:, ind], semdc[:, ind], 0.25, 'C2')
axins.axhline(y=tc_ref, color='k', linestyle='--')

axins.yaxis.set_ticks_position('right')
axins.yaxis.set_tick_params(labelsize=14)
axins.xaxis.set_tick_params(labelsize=14)
ax2.indicate_inset_zoom(axins)#, lw=0)
# fig.set_size_inches(10, 8)

# Labels and legends
ax1.set_ylabel(r'Response $R/\eta$', fontsize = '20')
ax2.set_xlabel('Time', fontsize = '20')
ax2.set_ylabel(r'Estimator $\widehat{\rho}$', fontsize = '20')
ax1.legend([l1, l2, l3], ['Equil.', 'Trans.', 'Subtr.'], fontsize='20', loc='best', ncols=3, columnspacing=0.8, handlelength=1)#, bbox_to_anchor=(0.5, -0.05))
ax2.xaxis.set_major_locator(MultipleLocator(0.5))
ax2.xaxis.set_minor_locator(MultipleLocator(0.25))
ax1.yaxis.set_tick_params(labelsize=16)
ax2.xaxis.set_tick_params(labelsize=16)
ax2.yaxis.set_tick_params(labelsize=16)

ax2.yaxis.set_major_locator(MultipleLocator(0.1))
ax1.yaxis.set_major_locator(MultipleLocator(0.2))#ticker.FixedLocator([-1, 0, 1]))
ax1.yaxis.set_minor_locator(MultipleLocator(0.1))

fig.align_ylabels()
plt.show()
# fig.savefig('LJ_shear_{}.pdf'.format(eta[ind]))

# %% ==========================================================================
# ########## 1D Langevin bias ##########
#==============================================================================
f = h5py.File('1d-lang_bias.jld2', 'r')
x = f['ηv'][:]
y1 = f['err'][:,0]/x
y2 = f['err'][:,1]/x

sl1 = np.polyfit(np.log10(x), np.log10(y1), 1)[0]
sl2 = np.polyfit(np.log10(x), np.log10(y2), 1)[0]

istep = 5
sl3 = np.log10(y1[-1]/y1[-1-istep])/np.log10(x[-1]/x[-1-istep])
sl4 = np.log10(y2[-1]/y2[-1-istep])/np.log10(x[-1]/x[-1-istep])

fig, ax = plt.subplots()

ax.loglog(x, y1, 'o', label=r"1$^\mathrm{st}$-order map $\Phi_\eta^1$", c="C0")
ax.loglog(x, y2, 's', label=r"2$^\mathrm{nd}$-order map $\Phi_\eta^2$", c="C1")

x_ref = np.linspace(min(x), max(x), 100)
C1 = y1[0]/x[0]
ax.loglog(x_ref, C1*x_ref, 'k-', label="reference (slope 1)")  # Black dashed line
ax.loglog(x_ref, x_ref**2, 'k--', label="reference (slope 2)")  # Black dash-dot line

# annotate with transform_rotates_text to align text and line
# ax.text(x[0], y1[0], r"1$^\mathrm{st}$-order map $\Phi_\eta^1$", color="blue", fontsize=12, rotation=45)
# ax.text(x[0], y2[0], r"2$^\mathrm{nd}$-order map $\Phi_\eta^2$",
        #  color="red", fontsize=12, rotation=33)

ax.set_xlabel(r'$\eta$', fontsize = '14')
ax.set_ylabel(r"Bias", fontsize = '14')
plt.legend(fontsize = '12')#, loc='upper left')#bbox_to_anchor=(1,1), loc="upper left")

plt.show()
# fig.savefig('1d-lang_bias.pdf')

# %% ==========================================================================
# ########## Variance plots (LJ mobility) ##########
#==============================================================================
f = h5py.File('upt_mobility_data.jld2', 'r')
R1, R2, dR, cR1, cR2, cdR, v1, v2, vd, cv1, cv2, cvd, tv, eta, M, sem1, sem2, semd, sem1c, sem2c, semdc = extract_values(f);

fig, (ax1, ax2, ax3) = plt.subplots(3)#, sharex=True)

l1, = ax1.plot(tv, cv2[:, 0])#/tv*eta[0]**2)#, label=r'$\eta = 0.01$')
l2, = ax1.plot(tv, cv2[:, 1])#/tv*eta[1]**2)#, label=r'$\eta = 0.1$')
l3, = ax1.plot(tv, cv2[:, 2])#/tv*eta[2]**2)#, label=r'$\eta = 1.0$')
ax2.plot(tv, cvd[:, 0])#/tv)#*eta[0]**2)
ax2.plot(tv, cvd[:, 1])#/tv)#*eta[1]**2)
ax2.plot(tv, cvd[:, 2])#/tv)#*eta[2]**2)

y1 = cvd[:, 0]/cvd[:, 1]
y2 = cvd[:, 1]/cvd[:, 2]

dec = 1500
tv2 = tv[dec:-1]
nn = len(tv2)
vals = np.linspace(tv[dec], tv[-1], num=nn) - tv[dec]

apt = 10
ax3.plot(tv[apt:-1], cv2[apt:-1,1]/cvd[apt:-1,1], label="0.1")#/tv)#*eta[1]**2)
ax3.plot(tv[apt:-1], cv2[apt:-1,2]/cvd[apt:-1,2], label="1.0")#/tv)#*eta[1]**2)
ax3.legend()
# ax3.plot(tv[1:-1], y2[1:-1])#/tv)#*eta[2]**2)
ax3.set_ylim(0, 100.0)
# ax3.set_ylim(0, 10)

# Labels and legends
# ax.set_ylabel(r'Variance$', fontsize = '14')
# ax.set_xlabel('Time', fontsize = '14')
# ax1.legend([l1, l2, l3], ['Equil.', 'Trans.', 'Subtr.'], fontsize='14', 
# loc='best', ncols=3)#, bbox_to_anchor=(0.5, -0.05))
ax1.legend([l1, l2, l3], ['0.01', '0.1', '1.0'],
           ncols=3,
        #    loc='upper right',
        # loc="best",
        #    bbox_to_anchor=(1.0, 0.9),
        #    shadow=True,
        #    borderaxespad=0.1,    # Small spacing around legend box
           title=r"Value of $\eta$"  # Title for the legend
           )

plt.subplots_adjust(right=0.85)
plt.show()

# %% ==========================================================================
# ########## Mock illustration: coupling measure ##########
#==============================================================================
fig, ax = plt.subplots()
plt.rcParams.update({
    'font.size': 8,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})
ax.spines[["top", "right"]].set_visible(False)

M = 10000;
eta = 0.05;
xLim = 1;
x11 = np.linspace(0.0, xLim-eta, M)
x22 = np.linspace(eta, xLim, M)

x = np.linspace(0.0, xLim, M);
xx = np.linspace(0.0, xLim, M);
x2 = xLim - x;
C1 = 0.1;
C2 = 10;
yy = x + C1/2*(np.cos(C2*x) + np.sin(C2*x));
y1 = yy + eta;
y2 = yy - eta;

px1 = (xLim - eta)/2;
px2 = (xLim + eta)/2;

no = int(M/2);
arl = 0.1;
pt1 = (px1, px1 + eta);
pt11 = (px1-arl, px1 + eta + arl);
pt2 = (px2, px2 - eta);
pt22 = (px2 + arl, px2 - eta - arl);


ptt1 = (x[-1], y1[-1]);
ptt10 = (ptt1[0], ptt1[1] + 0.15);
ptt2 = (x[-1], y2[-1]);
ptt20 = (ptt1[0], ptt2[1] - 0.15);

mv = 0.4;

etaoff = 0.02;
pteta = (ptt1[0] + etaoff, xLim - 5*etaoff);

ax.plot(x, y2, color = 'k')
ax.plot(x, y1, color = 'k')

ax.fill_between(x, y2, y1, fc="w", hatch="||")
plt.tick_params(top=False, bottom=False, left=False, right=False,
                labelleft=False, labelbottom=False)
ax.set_xlim(0, xLim+0.2)
ax.set_ylim(0, xLim+0.2)
ax.annotate(r'$O(\eta)$', xy=pteta, xytext=pteta, fontsize = '20')

ax.annotate('', xy=ptt1, xytext=ptt10, arrowprops=dict(arrowstyle='->'))#,rotation_mode='anchor',transform_rotates_text=True)
ax.annotate('', xy=ptt2, xytext=ptt20, arrowprops=dict(arrowstyle='->'))#,rotation_mode='anchor',transform_rotates_text=True)

ax.set_xlabel(r'$\mu$', fontsize = '16')
ax.set_ylabel(r'$\widetilde{\mu}_\eta$', fontsize = '16')
ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)  
ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

plt.show()
# fig.savefig('coup_meas_plot.eps', format='eps')