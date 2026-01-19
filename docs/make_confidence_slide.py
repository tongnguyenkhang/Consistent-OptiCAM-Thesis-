import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from textwrap import wrap

out_path = 'docs/confidence_conservation_slide.png'
# Slide size (landscape)
fig = plt.figure(figsize=(11, 6.1875), dpi=200)  # 2200x1237 px ~ 16:9-ish
ax = fig.add_axes([0,0,1,1])
ax.axis('off')

title = 'Confidence Conservation — main formulas & notation'
ax.text(0.5, 0.95, title, fontsize=26, ha='center', va='top', weight='bold')

left_x = 0.05
right_x = 0.55
y = 0.9
line_h = 0.06

left_block = [
    r'$x$ : image tensor $(3,H,W)$',
    r'$m_k$ : k^{th} mask, \, m_k(p)\in[0,1]$',
    r'$c_k$ : learned scalar (softmax-normalized)',
    r'$\odot$ : elementwise product (mask \odot image)'
]

for i, s in enumerate(left_block):
    ax.text(left_x, y - i*line_h, s, fontsize=16, va='top')

mid_y = 0.6
ax.text(left_x, mid_y, 'Union / residual:', fontsize=18, weight='semibold')
ax.text(left_x, mid_y - 0.04, r'$u(p)=\sum_{k=1}^K c_k\,m_k(p)$', fontsize=18, va='top')
ax.text(left_x, mid_y - 0.10, r'$r(p)=\mathrm{clamp}(1-u(p),0,1)$', fontsize=18, va='top')
ax.text(left_x, mid_y - 0.17, r'$x_{\mathrm{union}}=u\odot x\quad x_{\mathrm{res}}=r\odot x$', fontsize=16, va='top')

# Right column: scoring and losses
ax.text(right_x, 0.9, 'Scoring & canonical loss', fontsize=18, weight='semibold')
ax.text(right_x, 0.84, r'$s_{\mathrm{orig}}=f(x)$', fontsize=18, va='top')
ax.text(right_x, 0.78, r'$s_{\mathrm{union}}=f(x_{\mathrm{union}})$', fontsize=18, va='top')
ax.text(right_x, 0.72, r'$s_{\mathrm{res}}=f(x_{\mathrm{res}})$', fontsize=18, va='top')
ax.text(right_x, 0.64, r'Canonical reconstruction:', fontsize=16, va='top')
ax.text(right_x, 0.60, r'$s_{\mathrm{recon}}=f(x_{\mathrm{union}})+f(x_{\mathrm{res}})$', fontsize=18, va='top')

ax.text(right_x, 0.52, 'Canonical losses (per-sample):', fontsize=16, va='top')
ax.text(right_x, 0.48, r'Abs: $L=|s_{orig}-s_{recon}|$', fontsize=16, va='top')
ax.text(right_x, 0.44, r'Rel: $L=\frac{|s_{orig}-s_{recon}|}{|s_{orig}|+\epsilon}$', fontsize=16, va='top')
ax.text(right_x, 0.40, r'MSE: $L=(s_{orig}-s_{recon})^2$', fontsize=16, va='top')

# Bottom: short recommendations
bot_y = 0.22
ax.text(0.05, bot_y+0.08, 'Implementation notes (short):', fontsize=18, weight='semibold')
notes = [
    'Prefer probability-space scores (softmax) for canonical comparisons.',
    'Clamp denominators and residual mask: use small epsilon (1e-8).',
    'Optionally align optimizer objective: use s_union in loss (extra forward).',
    'Save both internal-loss history and canonical history per iteration.'
]
for i, n in enumerate(notes):
    ax.text(0.06, bot_y - i*0.035, '\u2022 ' + n, fontsize=14, va='top')

# footer
ax.text(0.5, 0.02, 'Generated: Confidence Conservation — compact slide', fontsize=10, ha='center')

plt.savefig(out_path, dpi=300)
print('Saved slide to', out_path)
