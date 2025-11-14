#!/usr/bin/env python3
"""
RabbitVideo System Architecture - Publication Quality Figure
Designed to match the theoretical depth of the paper while being visually elegant

Figure structure (single column, three panels stacked):
(a) Multi-tiered memory management with streaming execution
(b) LRU optimality: Sequential access → provably optimal
(c) Synchronous protocol: Eliminating ghost memory
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon, Wedge, Arc
from matplotlib.patches import ConnectionPatch, PathPatch
from matplotlib.path import Path
import numpy as np

# Professional academic styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'mathtext.fontset': 'stix',
    'axes.linewidth': 1.2,
    'lines.linewidth': 1.5,
})

# Sophisticated color palette (deeper, more professional)
COLORS = {
    'gpu': '#2E7D32',           # Deep green
    'cpu': '#1565C0',           # Deep blue
    'transfer': '#D84315',      # Deep red-orange
    'optimal': '#00695C',       # Dark teal
    'phase1': '#C62828',        # Deep red
    'phase2': '#EF6C00',        # Deep orange
    'phase3': '#6A1B9A',        # Deep purple
    'highlight': '#F9A825',     # Gold
    'text': '#212121',
    'text_light': '#616161',
    'border': '#424242',
}

# Create figure with three stacked panels
fig = plt.figure(figsize=(7, 9))  # Single column width
gs = fig.add_gridspec(3, 1, height_ratios=[1.4, 1, 1.2], hspace=0.4,
                      left=0.08, right=0.96, top=0.96, bottom=0.05)

# ============================================================================
# Panel (a): Multi-Tiered Memory Management with Streaming Execution
# ============================================================================
ax_a = fig.add_subplot(gs[0])
ax_a.set_xlim(0, 10)
ax_a.set_ylim(0, 10)
ax_a.axis('off')

# Panel label
ax_a.text(-0.5, 9.5, r'$\mathbf{(a)}$', fontsize=12, fontweight='bold')

# Title with key insight
ax_a.text(5, 9.5, r'Multi-Tiered Memory Management',
          fontsize=11, ha='center', fontweight='bold')
ax_a.text(5, 9.0, r'$\mathit{Working\ set}\ \mathcal{W}_t$ flows between GPU/CPU via streaming execution',
          fontsize=8, ha='center', style='italic', color=COLORS['text_light'])

# GPU Tier - more elegant design
gpu_main = FancyBboxPatch((0.5, 6.2), 9, 2.5,
                          boxstyle="round,pad=0.06",
                          edgecolor=COLORS['gpu'], facecolor='#E8F5E9',
                          linewidth=2, alpha=0.95)
ax_a.add_patch(gpu_main)

ax_a.text(5, 8.45, r'$\mathcal{M}_{\mathrm{GPU}} = 24\mathrm{GB}$',
          fontsize=10, ha='center', fontweight='bold', color=COLORS['gpu'])

# Working set with mathematical notation
working_set_box = FancyBboxPatch((1, 7.0), 8, 1.2,
                                 boxstyle="round,pad=0.04",
                                 edgecolor=COLORS['border'], facecolor='white',
                                 linewidth=1.2, linestyle='--', alpha=0.7)
ax_a.add_patch(working_set_box)

# Resident blocks - more compact and elegant
block_width = 1.45
for i in range(5):
    x = 1.4 + i * 1.6
    gradient_color = plt.cm.Greens(0.4 + i * 0.12)

    block = FancyBboxPatch((x, 7.2), block_width, 0.85,
                           boxstyle="round,pad=0.03",
                           edgecolor=COLORS['border'],
                           facecolor=gradient_color,
                           linewidth=1.3, alpha=0.9)
    ax_a.add_patch(block)

    ax_a.text(x + block_width/2, 7.75, r'$B_{%d}$' % i,
              ha='center', va='center', fontsize=9,
              fontweight='bold', color='white')
    ax_a.text(x + block_width/2, 7.4, '650MB',
              ha='center', va='center', fontsize=6.5,
              color='white', alpha=0.85)

ax_a.text(5, 6.65, r'$\mathcal{W}_t = \{B_0, B_1, ..., B_4\}$ (working set: $k=5$ blocks)',
          fontsize=7.5, ha='center', style='italic', color=COLORS['text_light'])

# Memory partitions below - more sophisticated layout
partitions = [
    (0.8, 6.3, 2.7, 0.35, r'$M_{\mathrm{act}}$', '#7B1FA2', 'Activations'),
    (3.7, 6.3, 2.5, 0.35, r'$M_{\mathrm{aux}}$', '#E65100', 'VAE/Enc'),
    (6.4, 6.3, 2.8, 0.35, r'$M_{\mathrm{buffer}}$', '#1565C0', 'Buffers'),
]

for x, y, w, h, var, color, label in partitions:
    rect = Rectangle((x, y), w, h,
                     edgecolor=color, facecolor=color,
                     linewidth=1.2, alpha=0.35)
    ax_a.add_patch(rect)
    ax_a.text(x + w/2, y + h/2, f'{var}',
              ha='center', va='center', fontsize=7,
              color=color, fontweight='bold')

# Memory invariant - prominent mathematical formula
invariant_box = FancyBboxPatch((0.5, 5.7), 9, 0.42,
                               boxstyle="round,pad=0.03",
                               edgecolor=COLORS['optimal'], facecolor='#E0F2F1',
                               linewidth=1.8, alpha=0.95)
ax_a.add_patch(invariant_box)
ax_a.text(5, 5.91,
          r'$\mathbf{Invariant:}\ \ \sum_{B_i \in \mathcal{W}_t} \mathrm{size}(B_i) + M_{\mathrm{act}} + M_{\mathrm{aux}} \leq M_{\mathrm{GPU}}$',
          fontsize=7.5, ha='center', fontweight='bold', color=COLORS['optimal'])

# PCIe Transfer Bus - more dynamic visualization
pcie_y = 5.0
# Create flowing effect
for i in range(35):
    phase = i * 2 * np.pi / 35
    alpha_val = 0.25 + 0.2 * (np.sin(phase) + 1) / 2
    width_val = 6 + 2 * np.sin(phase + np.pi/4)
    ax_a.plot([0.5 + i*0.265, 0.765 + i*0.265], [pcie_y, pcie_y],
              color=COLORS['transfer'], linewidth=width_val,
              alpha=alpha_val, solid_capstyle='round')

# PCIe label with transfer rate
ax_a.text(5, pcie_y, r'$\mathbf{PCIe\ Gen4\ \times16}$  |  $\tau_{\mathrm{xfer}} \approx 100\mathrm{ms/block}$',
          fontsize=8, ha='center', fontweight='bold', color='white',
          bbox=dict(boxstyle='round,pad=0.22', facecolor=COLORS['transfer'],
                   alpha=0.92, edgecolor=COLORS['border'], linewidth=1.3))

# Bidirectional arrows with labels
# Upward (Load)
arrow_up = FancyArrowPatch((3.2, 4.3), (3.2, 5.6),
                          arrowstyle='->', mutation_scale=18,
                          linewidth=2.2, color=COLORS['phase2'],
                          alpha=0.8, zorder=5)
ax_a.add_patch(arrow_up)
ax_a.text(2.65, 4.95, r'$\mathrm{Load}$', fontsize=7,
          color=COLORS['phase2'], fontweight='bold', rotation=90, va='center')

# Downward (Evict)
arrow_down = FancyArrowPatch((6.8, 5.6), (6.8, 4.3),
                            arrowstyle='->', mutation_scale=18,
                            linewidth=2.2, color=COLORS['phase1'],
                            alpha=0.8, zorder=5)
ax_a.add_patch(arrow_down)
ax_a.text(7.35, 4.95, r'$\mathrm{Evict}$', fontsize=7,
          color=COLORS['phase1'], fontweight='bold', rotation=90, va='center')

# CPU Tier - elegant design
cpu_main = FancyBboxPatch((0.5, 0.3), 9, 3.5,
                         boxstyle="round,pad=0.06",
                         edgecolor=COLORS['cpu'], facecolor='#E3F2FD',
                         linewidth=2, alpha=0.95)
ax_a.add_patch(cpu_main)

ax_a.text(5, 3.55, r'$\mathcal{M}_{\mathrm{CPU}}$ (System DRAM)',
          fontsize=10, ha='center', fontweight='bold', color=COLORS['cpu'])

# Offloaded blocks - compact grid with visual appeal
offload_box = FancyBboxPatch((0.9, 0.6), 8.2, 2.7,
                             boxstyle="round,pad=0.04",
                             edgecolor=COLORS['border'], facecolor='white',
                             linewidth=1.2, linestyle='--', alpha=0.5)
ax_a.add_patch(offload_box)

ax_a.text(5, 3.15, r'$\mathcal{C} = \{B_5, ..., B_{59}\}$ (offloaded: $N-k=55$ blocks)',
          fontsize=7.5, ha='center', style='italic', color=COLORS['text_light'])

# Compact grid of offloaded blocks
rows, cols = 5, 11
bw, bh = 0.7, 0.38
sx, sy = 1.2, 0.8

for row in range(rows):
    for col in range(cols):
        idx = row * cols + col
        if idx >= 55:
            break
        x = sx + col * (bw + 0.04)
        y = sy + row * (bh + 0.08)

        intensity = 0.3 + 0.45 * (idx / 54)
        rect = Rectangle((x, y), bw, bh,
                        edgecolor='#90A4AE', facecolor=plt.cm.Blues(intensity),
                        linewidth=0.7, alpha=0.8)
        ax_a.add_patch(rect)

        # Label key blocks
        if idx < 2 or idx >= 53:
            ax_a.text(x + bw/2, y + bh/2, f'{idx+5}',
                     ha='center', va='center', fontsize=5,
                     color='white', fontweight='bold')

# System components - elegant annotation at bottom
components = [
    ('BlockTracker', r'$O(k\log k)$ eviction'),
    ('BlockManager', 'Sync protocol'),
    ('MemoryMonitor', 'State tracking'),
]

comp_y = 0.45
for i, (name, desc) in enumerate(components):
    x = 1.2 + i * 2.9
    ax_a.text(x, comp_y, r'$\bullet$', fontsize=10, color=COLORS['optimal'], fontweight='bold')
    ax_a.text(x + 0.2, comp_y, f'{name}', fontsize=7, fontweight='bold', color=COLORS['text'])
    ax_a.text(x + 0.2, comp_y - 0.15, desc, fontsize=6, style='italic', color=COLORS['text_light'])

# ============================================================================
# Panel (b): LRU Optimality - Sequential Access Pattern
# ============================================================================
ax_b = fig.add_subplot(gs[1])
ax_b.set_xlim(0, 10)
ax_b.set_ylim(0, 6.5)
ax_b.axis('off')

# Panel label
ax_b.text(-0.5, 6.2, r'$\mathbf{(b)}$', fontsize=12, fontweight='bold')

# Title emphasizing the key insight
ax_b.text(5, 6.2, r'Sequential Access $\Rightarrow$ LRU Optimality',
          fontsize=11, ha='center', fontweight='bold')

# Theorem box with elegant styling
theorem_box = FancyBboxPatch((0.3, 4.6), 9.4, 1.35,
                            boxstyle="round,pad=0.05",
                            edgecolor=COLORS['optimal'], facecolor='#E0F7FA',
                            linewidth=1.8, alpha=0.95)
ax_b.add_patch(theorem_box)

# Theorem statement
theorem_text = (r'$\mathbf{Theorem}$ (Sequential Optimality): ' +
                'For sequential access $B_1 \!\rightarrow\! B_2 \!\rightarrow\! \cdots \!\rightarrow\! B_N$,' + '\n' +
                r'LRU achieves clairvoyant optimum: $\mathrm{Transfers}_{\mathrm{LRU}} = \mathrm{Transfers}_{\mathrm{OPT}} = T \cdot \max(0, N\!-\!k)$')
ax_b.text(5, 5.27, theorem_text, ha='center', va='center',
          fontsize=7.8, color=COLORS['optimal'],
          bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                   edgecolor=COLORS['optimal'], linewidth=1.2, alpha=0.85))

# Sequential access pattern visualization
ax_b.text(0.5, 4.1, r'$\mathbf{Access\ pattern:}$', fontsize=8.5,
          fontweight='bold', color=COLORS['text'])

# Blocks with flowing arrows
y = 3.5
block_colors = plt.cm.Oranges(np.linspace(0.4, 0.85, 8))
for i in range(8):
    x = 0.8 + i * 1.1

    rect = Rectangle((x, y-0.22), 0.85, 0.44,
                    edgecolor=COLORS['border'], facecolor=block_colors[i],
                    linewidth=1.2, alpha=0.9)
    ax_b.add_patch(rect)
    ax_b.text(x + 0.425, y, r'$B_{%d}$' % i,
              ha='center', va='center', fontsize=7.5,
              fontweight='bold', color='white')

    # Sequential arrows
    if i < 7:
        arrow = FancyArrowPatch((x + 0.85, y), (x + 1.1, y),
                               arrowstyle='->', mutation_scale=14,
                               linewidth=2, color=COLORS['transfer'],
                               alpha=0.7)
        ax_b.add_patch(arrow)

ax_b.text(9.6, y, r'$\cdots$', ha='center', va='center',
          fontsize=13, fontweight='bold', color=COLORS['text_light'])

# Strategy comparison - clean and clear
ax_b.text(0.5, 2.5, r'$\mathbf{Eviction\ strategies:}$', fontsize=8.5,
          fontweight='bold', color=COLORS['text'])

strategies = [
    (r'$\mathbf{LRU}$', r'31 transfers', COLORS['optimal'], True, 1.5),
    (r'$\mathbf{Random}$', r'82 transfers', '#C62828', False, 4.2),
    (r'$\mathbf{Optimal}$', r'31 transfers', COLORS['optimal'], True, 6.9),
]

for name, result, color, is_optimal, x in strategies:
    y = 1.7
    box = FancyBboxPatch((x, y), 2.4, 0.65,
                        boxstyle="round,pad=0.04",
                        edgecolor=color,
                        facecolor=color if is_optimal else '#FFEBEE',
                        linewidth=1.8 if is_optimal else 1.2,
                        alpha=0.9 if is_optimal else 0.5)
    ax_b.add_patch(box)

    text_color = 'white' if is_optimal else color
    ax_b.text(x + 1.2, y + 0.42, name, ha='center', va='center',
              fontsize=8, color=text_color, fontweight='bold')
    ax_b.text(x + 1.2, y + 0.18, result, ha='center', va='center',
              fontsize=7, color=text_color)

# Equivalence annotation
ax_b.annotate('', xy=(6.9, 1.35), xytext=(3.9, 1.35),
             arrowprops=dict(arrowstyle='<->', lw=2.5,
                           color=COLORS['optimal'], alpha=0.7))
ax_b.text(5.4, 1.05, r'$\mathbf{Provably\ equivalent}$',
          ha='center', fontsize=7.5, fontweight='bold',
          color=COLORS['optimal'],
          bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                   edgecolor=COLORS['optimal'], linewidth=1.2, alpha=0.9))

# Efficiency metric
ax_b.text(5, 0.5, r'$\eta_{\mathrm{transfer}} = 85\%$ PCIe utilization  |  $2.6\times$ better than random',
          ha='center', fontsize=7, style='italic',
          color=COLORS['text_light'],
          bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFFDE7',
                   edgecolor=COLORS['highlight'], linewidth=1, alpha=0.7))

# ============================================================================
# Panel (c): Synchronous Protocol - Eliminating Ghost Memory
# ============================================================================
ax_c = fig.add_subplot(gs[2])
ax_c.set_xlim(0, 10)
ax_c.set_ylim(0, 8)
ax_c.axis('off')

# Panel label
ax_c.text(-0.5, 7.7, r'$\mathbf{(c)}$', fontsize=12, fontweight='bold')

# Title
ax_c.text(5, 7.7, r'Synchronous Protocol: Eliminating Ghost Memory',
          fontsize=11, ha='center', fontweight='bold')
ax_c.text(5, 7.25, r'$\mathit{Double\ cache\ invalidation}$ ensures deterministic memory reclamation',
          fontsize=8, ha='center', style='italic', color=COLORS['text_light'])

# Three-phase protocol with clean design
phases = [
    {
        'num': 1,
        'name': 'Atomic Transfer',
        'color': COLORS['phase1'],
        'y': 5.8,
        'steps': [
            r'$B_i.\mathrm{device} \leftarrow \mathrm{target}$',
            r'$\textsc{Barrier}()$ // enforce completion',
        ]
    },
    {
        'num': 2,
        'name': 'Memory Reclamation',
        'color': COLORS['phase2'],
        'y': 3.9,
        'steps': [
            r'$\textsc{InvalidateCache}()$ // $1^{\mathrm{st}}$ release',
            r'$\textsc{Barrier}()$ // sync deallocation',
            r'$\textsc{InvalidateCache}()$ // $2^{\mathrm{nd}}$ consolidate',
        ]
    },
    {
        'num': 3,
        'name': 'State Verification',
        'color': COLORS['phase3'],
        'y': 1.7,
        'steps': [
            r'$\mathcal{W}_t \leftarrow \mathcal{W}_t \setminus \{B_i\}$',
            r'$\mathbf{assert}\ M_{\mathrm{reserved}} - M_{\mathrm{allocated}} < \epsilon$',
        ]
    }
]

for phase in phases:
    y = phase['y']
    color = phase['color']

    # Phase header with number badge
    header = FancyBboxPatch((0.5, y + 0.85), 9, 0.42,
                           boxstyle="round,pad=0.03",
                           edgecolor=color, facecolor=color,
                           linewidth=1.6, alpha=0.92)
    ax_c.add_patch(header)

    # Phase number circle
    circle = Circle((0.9, y + 1.06), 0.16,
                   edgecolor='white', facecolor='white',
                   linewidth=1.2, zorder=10)
    ax_c.add_patch(circle)
    ax_c.text(0.9, y + 1.06, str(phase['num']),
             ha='center', va='center', fontsize=8.5,
             color=color, fontweight='bold', zorder=11)

    # Phase name
    ax_c.text(1.3, y + 1.06, r'$\mathbf{Phase\ ' + str(phase['num']) + ':}$ ' + phase['name'],
             ha='left', va='center', fontsize=8.5,
             color='white', fontweight='bold')

    # Steps container
    step_box = FancyBboxPatch((0.5, y), 9, 0.75,
                             boxstyle="round,pad=0.03",
                             edgecolor=color, facecolor='white',
                             linewidth=1.2, alpha=0.4)
    ax_c.add_patch(step_box)

    # Steps
    for i, step in enumerate(phase['steps']):
        step_y = y + 0.58 - i * 0.22
        ax_c.text(0.9, step_y, step,
                 ha='left', va='center', fontsize=7,
                 color=COLORS['text'], family='monospace')

# Timing guarantee at bottom
timing_box = FancyBboxPatch((0.5, 0.5), 9, 0.55,
                           boxstyle="round,pad=0.03",
                           edgecolor=COLORS['transfer'], facecolor='#FFF3E0',
                           linewidth=1.5, alpha=0.95)
ax_c.add_patch(timing_box)

ax_c.text(5, 0.9,
         r'$\mathbf{Latency\ bound:}$ $\tau_{\mathrm{total}} = \tau_{\mathrm{transfer}} + \tau_{\mathrm{sync}} \approx 100\mathrm{ms} + 10\mathrm{ms}$',
         ha='center', va='center', fontsize=7.5,
         color=COLORS['transfer'], fontweight='bold')
ax_c.text(5, 0.62,
         r'$\mathbf{Guarantee:}$ Ghost memory eliminated; $M_{\mathrm{reserved}}$ freed synchronously',
         ha='center', va='center', fontsize=7,
         color=COLORS['text_light'], style='italic')

# ============================================================================
# Final touches
# ============================================================================
plt.tight_layout()

# Save in multiple formats
output_pdf = 'rabbitvideo_system_architecture.pdf'
output_png = 'rabbitvideo_system_architecture.png'

plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
plt.savefig(output_png, dpi=300, bbox_inches='tight', format='png')

print("=" * 80)
print("✓ RabbitVideo System Architecture Figure Generated")
print("=" * 80)
print(f"Output files:")
print(f"  • {output_pdf} (vector, for paper)")
print(f"  • {output_png} (raster, for preview)")
print()
print("Figure design highlights:")
print("  • Panel (a): Multi-tiered memory with streaming execution model")
print("  • Panel (b): LRU optimality proof for sequential access")
print("  • Panel (c): Synchronous protocol eliminating ghost memory")
print()
print("Key visual features:")
print("  • Mathematical rigor: Formal notation and theorem statements")
print("  • Clean layout: Single-column format for paper")
print("  • Professional styling: Academic color palette and typography")
print("  • Information density: High but balanced for clarity")
print("=" * 80)

plt.show()
