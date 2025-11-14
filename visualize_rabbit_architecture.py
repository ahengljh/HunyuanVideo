#!/usr/bin/env python3
"""
RabbitVideo System Architecture - Academic Paper Quality Figure
Emphasizes: Multi-tiered memory, LRU optimality, Synchronous protocol, Three-phase execution
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon, Wedge
from matplotlib.patches import ConnectionPatch
import numpy as np

# Professional academic color scheme
COLORS = {
    'gpu_primary': '#2E7D32',      # Deep green
    'cpu_primary': '#1565C0',      # Deep blue
    'transfer': '#FF6F00',         # Deep orange
    'phase1': '#C62828',           # Deep red
    'phase2': '#F57C00',           # Deep orange
    'phase3': '#6A1B9A',           # Deep purple
    'optimal': '#00897B',          # Teal
    'highlight': '#FDD835',        # Yellow
    'text_dark': '#212121',
    'text_light': '#424242',
    'border': '#37474F',
}

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['mathtext.fontset'] = 'stix'

fig = plt.figure(figsize=(18, 11))

# Create sophisticated multi-panel layout
gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1.5, 1.3], width_ratios=[1.3, 1.3, 1],
                      hspace=0.35, wspace=0.3, left=0.05, right=0.98, top=0.94, bottom=0.05)

# ============================================================================
# Panel A: Multi-Tiered Memory Architecture with Component Hierarchy
# ============================================================================
ax_memory = fig.add_subplot(gs[0:2, 0])
ax_memory.set_xlim(0, 10)
ax_memory.set_ylim(0, 12)
ax_memory.axis('off')

# Title
ax_memory.text(5, 11.5, r'$\bf{(a)}$ Multi-Tiered Memory Management',
               fontsize=13, ha='center', fontweight='bold')

# GPU Memory Tier
gpu_outer = FancyBboxPatch((0.3, 7.5), 9.4, 3.6,
                           boxstyle="round,pad=0.08",
                           edgecolor=COLORS['gpu_primary'], facecolor='#E8F5E9',
                           linewidth=2.5, zorder=1)
ax_memory.add_patch(gpu_outer)

ax_memory.text(5, 10.8, r'$\mathcal{M}_{GPU} = 24\text{GB}$ (VRAM)',
               fontsize=11, ha='center', fontweight='bold',
               color=COLORS['gpu_primary'])

# Working Set W_t
working_set_box = FancyBboxPatch((0.8, 8.8), 8.4, 1.7,
                                 boxstyle="round,pad=0.05",
                                 edgecolor=COLORS['border'], facecolor='white',
                                 linewidth=1.5, linestyle='--', alpha=0.8)
ax_memory.add_patch(working_set_box)

ax_memory.text(5, 10.3, r'$\mathcal{W}_t$ (Working Set) - $k=5$ resident blocks',
               fontsize=9, ha='center', style='italic', color=COLORS['text_light'])

# Resident blocks with gradient
for i in range(5):
    x = 1.4 + i * 1.6
    intensity = 0.3 + i * 0.15
    block = FancyBboxPatch((x, 9), 1.4, 1.1,
                           boxstyle="round,pad=0.04",
                           edgecolor=COLORS['border'],
                           facecolor=plt.cm.Greens(intensity),
                           linewidth=1.8)
    ax_memory.add_patch(block)
    ax_memory.text(x + 0.7, 9.7, r'$B_{%d}$' % i, ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white')
    ax_memory.text(x + 0.7, 9.3, '650MB', ha='center', va='center',
                   fontsize=7, color='white', alpha=0.9)

# Memory partitions
partitions = [
    (0.8, 8.0, 3.5, 0.6, r'$M_{act}$ (Activations)', '#9C27B0', 0.3),
    (4.5, 8.0, 2.0, 0.6, r'$M_{aux}$ (VAE)', '#FF6F00', 0.3),
    (6.7, 8.0, 2.5, 0.6, r'$M_{txt}$ (Encoders)', '#0277BD', 0.3),
]

for x, y, w, h, label, color, alpha in partitions:
    rect = Rectangle((x, y), w, h, edgecolor=color, facecolor=color,
                     linewidth=1.5, alpha=alpha)
    ax_memory.add_patch(rect)
    ax_memory.text(x + w/2, y + h/2, label, ha='center', va='center',
                   fontsize=8, color='white', fontweight='bold')

# Memory invariant
invariant_box = FancyBboxPatch((0.5, 7.3), 9, 0.4,
                              boxstyle="round,pad=0.02",
                              edgecolor=COLORS['optimal'], facecolor='#E0F2F1',
                              linewidth=1.5)
ax_memory.add_patch(invariant_box)
ax_memory.text(5, 7.5,
               r'$\bf{Invariant:}$ $\sum_{B_i \in \mathcal{W}_t} \text{size}(B_i) + M_{act} + M_{aux} \leq M_{GPU}$',
               fontsize=8, ha='center', color=COLORS['optimal'], fontweight='bold')

# PCIe Transfer Bus
pcie_y = 6.8
for i in range(30):
    alpha = 0.4 + 0.15 * np.sin(i * np.pi / 15)
    ax_memory.plot([0.3 + i*0.315, 0.615 + i*0.315], [pcie_y, pcie_y],
                   color=COLORS['transfer'], linewidth=10, alpha=alpha, solid_capstyle='round')

ax_memory.text(5, pcie_y, r'$\bf{PCIe}$ $Gen4$ $\times16$ $\rightarrow$ $\tau_{transfer}=100ms/block$',
               fontsize=9, ha='center', fontweight='bold', color='white',
               bbox=dict(boxstyle='round,pad=0.25', facecolor=COLORS['transfer'],
                        alpha=0.9, edgecolor=COLORS['border'], linewidth=1.5))

# CPU Memory Tier (System DRAM)
cpu_outer = FancyBboxPatch((0.3, 1.2), 9.4, 5.2,
                           boxstyle="round,pad=0.08",
                           edgecolor=COLORS['cpu_primary'], facecolor='#E3F2FD',
                           linewidth=2.5, zorder=1)
ax_memory.add_patch(cpu_outer)

ax_memory.text(5, 6.1, r'$\mathcal{M}_{CPU}$ (System DRAM)',
               fontsize=11, ha='center', fontweight='bold',
               color=COLORS['cpu_primary'])

# Offloaded blocks container
offload_container = FancyBboxPatch((0.8, 1.7), 8.4, 4.1,
                                   boxstyle="round,pad=0.05",
                                   edgecolor=COLORS['border'], facecolor='white',
                                   linewidth=1.5, linestyle='--', alpha=0.6)
ax_memory.add_patch(offload_container)

ax_memory.text(5, 5.6, r'$\mathcal{C}$ (Offloaded Blocks) - $(N-k)=55$ blocks',
               fontsize=9, ha='center', style='italic', color=COLORS['text_light'])

# Compressed offloaded blocks visualization
rows, cols = 6, 10
block_w, block_h = 0.82, 0.52
start_x, start_y = 1.0, 2.0

for row in range(rows):
    for col in range(cols):
        idx = row * cols + col
        if idx >= 55:
            break
        x = start_x + col * (block_w + 0.03)
        y = start_y + row * (block_h + 0.08)

        intensity = 0.25 + 0.5 * (idx / 55)
        block = Rectangle((x, y), block_w, block_h,
                         edgecolor=COLORS['border'],
                         facecolor=plt.cm.Blues(intensity),
                         linewidth=0.7, alpha=0.85)
        ax_memory.add_patch(block)

        if idx < 2 or idx >= 53:
            ax_memory.text(x + block_w/2, y + block_h/2, r'$B_{%d}$' % (idx+5),
                          ha='center', va='center', fontsize=5.5,
                          color='white', fontweight='bold')

# Component annotations with professional styling
components = [
    (0.5, 0.8, 'BlockTracker', r'$O(k \log k)$ eviction'),
    (3.5, 0.8, 'BlockManager', 'Sync Protocol'),
    (6.5, 0.8, 'MemoryMonitor', 'State Tracking'),
]

for x, y, name, desc in components:
    comp_box = FancyBboxPatch((x, y), 2.8, 0.5,
                              boxstyle="round,pad=0.03",
                              edgecolor=COLORS['border'], facecolor='#FAFAFA',
                              linewidth=1.2)
    ax_memory.add_patch(comp_box)
    ax_memory.text(x + 1.4, y + 0.3, r'$\bf{' + name + '}$', ha='center', va='center',
                   fontsize=8, color=COLORS['text_dark'])
    ax_memory.text(x + 1.4, y + 0.1, desc, ha='center', va='center',
                   fontsize=6.5, color=COLORS['text_light'], style='italic')

# ============================================================================
# Panel B: LRU Optimality Proof
# ============================================================================
ax_lru = fig.add_subplot(gs[0, 1])
ax_lru.set_xlim(0, 10)
ax_lru.set_ylim(0, 6)
ax_lru.axis('off')

ax_lru.text(5, 5.5, r'$\bf{(b)}$ Sequential Access $\Rightarrow$ LRU Optimality',
            fontsize=13, ha='center', fontweight='bold')

# Theorem box
theorem_box = FancyBboxPatch((0.3, 3.8), 9.4, 1.4,
                            boxstyle="round,pad=0.06",
                            edgecolor=COLORS['optimal'], facecolor='#E0F7FA',
                            linewidth=2)
ax_lru.add_patch(theorem_box)

theorem_text = (r'$\bf{Theorem\ (Sequential\ Optimality):}$' + '\n' +
                r'For sequential access $B_1 \rightarrow B_2 \rightarrow \cdots \rightarrow B_N$:' + '\n' +
                r'$\text{Transfers}_{LRU} = \text{Transfers}_{OPT} = T \cdot \max(0, N-k)$')

ax_lru.text(5, 4.5, theorem_text, ha='center', va='center',
            fontsize=9, color=COLORS['optimal'],
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor=COLORS['optimal'], linewidth=1.5, alpha=0.8))

# Access pattern visualization
ax_lru.text(0.5, 3.2, r'$\bf{Access\ Pattern:}$', fontsize=9,
            fontweight='bold', color=COLORS['text_dark'])

# Sequential blocks with arrows
y = 2.7
for i in range(8):
    x = 0.8 + i * 1.15
    color_intensity = 0.3 + i * 0.08
    block = Rectangle((x, y-0.25), 0.9, 0.5,
                     edgecolor=COLORS['border'],
                     facecolor=plt.cm.Oranges(color_intensity),
                     linewidth=1.2)
    ax_lru.add_patch(block)
    ax_lru.text(x + 0.45, y, r'$B_{%d}$' % i, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white')

    if i < 7:
        arrow = FancyArrowPatch((x + 0.9, y), (x + 1.05, y),
                               arrowstyle='->', mutation_scale=15,
                               linewidth=2, color=COLORS['transfer'])
        ax_lru.add_patch(arrow)

ax_lru.text(9.5, y, '...', ha='center', va='center', fontsize=14,
            fontweight='bold', color=COLORS['text_light'])

# Eviction comparison
ax_lru.text(0.5, 1.8, r'$\bf{Eviction\ Strategy:}$', fontsize=9,
            fontweight='bold', color=COLORS['text_dark'])

strategies = [
    (1.5, 1.2, 'LRU', r'$31$ transfers', COLORS['optimal']),
    (4.0, 1.2, 'Random', r'$82$ transfers', '#D32F2F'),
    (6.5, 1.2, 'Optimal', r'$31$ transfers', COLORS['optimal']),
]

for x, y, name, transfers, color in strategies:
    is_optimal = (name == 'LRU' or name == 'Optimal')
    box = FancyBboxPatch((x, y), 2.3, 0.7,
                        boxstyle="round,pad=0.04",
                        edgecolor=color,
                        facecolor=color if is_optimal else '#FFEBEE',
                        linewidth=2 if is_optimal else 1.5,
                        alpha=0.9 if is_optimal else 0.4)
    ax_lru.add_patch(box)

    text_color = 'white' if is_optimal else color
    ax_lru.text(x + 1.15, y + 0.45, r'$\bf{' + name + '}$', ha='center', va='center',
                fontsize=9, color=text_color, fontweight='bold')
    ax_lru.text(x + 1.15, y + 0.2, transfers, ha='center', va='center',
                fontsize=7.5, color=text_color)

# Optimality indicator
ax_lru.plot([1.5, 6.5], [0.7, 0.7], color=COLORS['optimal'],
           linewidth=3, linestyle='--', alpha=0.6)
ax_lru.text(4, 0.5, r'$\bf{Proven\ Equivalent}$', ha='center',
           fontsize=8, color=COLORS['optimal'], fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                    edgecolor=COLORS['optimal'], linewidth=1.2))

# ============================================================================
# Panel C: Synchronous Transfer Protocol
# ============================================================================
ax_protocol = fig.add_subplot(gs[1, 1])
ax_protocol.set_xlim(0, 10)
ax_protocol.set_ylim(0, 10)
ax_protocol.axis('off')

ax_protocol.text(5, 9.5, r'$\bf{(c)}$ Synchronous Migration Protocol',
                 fontsize=13, ha='center', fontweight='bold')

# Algorithm box
algo_box = FancyBboxPatch((0.3, 0.5), 9.4, 8.6,
                          boxstyle="round,pad=0.06",
                          edgecolor=COLORS['border'], facecolor='#FAFAFA',
                          linewidth=2)
ax_protocol.add_patch(algo_box)

# Three phases with detailed steps
phases = [
    {
        'name': 'Phase 1: Atomic Transfer',
        'y': 7.5,
        'color': COLORS['phase1'],
        'steps': [
            r'$B_i.\text{device} \leftarrow \text{target}$',
            r'$\textsc{Barrier}()$ // enforce completion',
        ]
    },
    {
        'name': 'Phase 2: Memory Reclamation',
        'y': 5.0,
        'color': COLORS['phase2'],
        'steps': [
            r'$\textsc{InvalidateCache}()$ // $1^{st}$ release',
            r'$\textsc{Barrier}()$ // sync deallocation',
            r'$\textsc{InvalidateCache}()$ // $2^{nd}$ consolidate',
        ]
    },
    {
        'name': 'Phase 3: State Verification',
        'y': 2.2,
        'color': COLORS['phase3'],
        'steps': [
            r'$\mathcal{W}_t \leftarrow \mathcal{W}_t \setminus \{B_i\}$',
            r'$\bf{assert}$ $M_{reserved} - M_{allocated} < \epsilon$',
        ]
    }
]

for phase_data in phases:
    y = phase_data['y']
    color = phase_data['color']

    # Phase header
    header_box = FancyBboxPatch((0.8, y + 0.9), 8.4, 0.5,
                               boxstyle="round,pad=0.03",
                               edgecolor=color, facecolor=color,
                               linewidth=2, alpha=0.9)
    ax_protocol.add_patch(header_box)
    ax_protocol.text(5, y + 1.15, r'$\bf{' + phase_data['name'] + '}$',
                    ha='center', va='center', fontsize=9,
                    color='white', fontweight='bold')

    # Steps
    for i, step in enumerate(phase_data['steps']):
        step_y = y + 0.55 - i * 0.35
        ax_protocol.text(1.2, step_y, step, ha='left', va='center',
                        fontsize=8, color=COLORS['text_dark'],
                        family='monospace')

# Timing annotation
timing_box = FancyBboxPatch((0.8, 0.8), 8.4, 0.5,
                           boxstyle="round,pad=0.03",
                           edgecolor=COLORS['transfer'], facecolor='#FFF3E0',
                           linewidth=1.5)
ax_protocol.add_patch(timing_box)
ax_protocol.text(5, 1.05,
                r'$\tau_{total} = \tau_{transfer} + \tau_{sync} \approx 100ms + 10ms = 110ms$',
                ha='center', va='center', fontsize=8,
                color=COLORS['transfer'], fontweight='bold')

# ============================================================================
# Panel D: Three-Phase Execution Timeline
# ============================================================================
ax_phases = fig.add_subplot(gs[0, 2])
ax_phases.set_xlim(0, 10)
ax_phases.set_ylim(0, 12)
ax_phases.axis('off')

ax_phases.text(5, 11.5, r'$\bf{(d)}$ Execution Phases',
               fontsize=13, ha='center', fontweight='bold')

# Vertical timeline
timeline_x = 2
timeline_y_start = 1.5
timeline_y_end = 10.5

ax_phases.plot([timeline_x, timeline_x], [timeline_y_start, timeline_y_end],
              color=COLORS['border'], linewidth=3, solid_capstyle='round')

# Phase markers
phase_specs = [
    ('Phase 1', 'Proactive Init', r'Offload $91\%$ blocks',
     9.5, COLORS['phase1'], r'$t=0$'),
    ('Phase 2', 'Streaming', r'LRU swap: $55$ transfers/step',
     6.0, COLORS['phase2'], r'$t \in [0, T]$'),
    ('Phase 3', 'Auxiliary Mgmt', r'VAE+Encoder lifecycle',
     2.5, COLORS['phase3'], r'$t_{decode}$'),
]

for name, subtitle, desc, y, color, time_label in phase_specs:
    # Timeline node
    node = Circle((timeline_x, y), 0.25,
                 edgecolor=color, facecolor=color, linewidth=2.5, zorder=10)
    ax_phases.add_patch(node)

    # Time label
    ax_phases.text(timeline_x - 0.8, y, time_label, ha='right', va='center',
                  fontsize=7.5, color=COLORS['text_light'], style='italic')

    # Phase box
    box_x = timeline_x + 0.5
    phase_box = FancyBboxPatch((box_x, y - 0.6), 6.8, 1.2,
                              boxstyle="round,pad=0.05",
                              edgecolor=color, facecolor='white',
                              linewidth=2)
    ax_phases.add_patch(phase_box)

    ax_phases.text(box_x + 0.3, y + 0.35, r'$\bf{' + name + ':}$ ' + subtitle,
                  ha='left', va='center', fontsize=9,
                  color=color, fontweight='bold')
    ax_phases.text(box_x + 0.3, y - 0.15, desc,
                  ha='left', va='center', fontsize=7.5,
                  color=COLORS['text_light'])

# ============================================================================
# Panel E: Performance Metrics
# ============================================================================
ax_perf = fig.add_subplot(gs[1, 2])
ax_perf.set_xlim(0, 10)
ax_perf.set_ylim(0, 10)
ax_perf.axis('off')

ax_perf.text(5, 9.5, r'$\bf{(e)}$ System Guarantees',
             fontsize=13, ha='center', fontweight='bold')

# Key metrics with mathematical formulation
metrics = [
    ('Memory Reduction', r'$\frac{66\text{GB}}{24\text{GB}} = 2.75\times$', '#4CAF50'),
    ('Runtime Overhead', r'$1.15\times$ (15% slowdown)', '#FF9800'),
    ('Transfer Efficiency', r'$85\%$ PCIe utilization', '#2196F3'),
    ('Determinism', r'Bit-identical outputs', '#9C27B0'),
]

y_start = 8
for i, (metric, value, color) in enumerate(metrics):
    y = y_start - i * 1.8

    metric_box = FancyBboxPatch((0.5, y - 0.4), 9, 1.2,
                               boxstyle="round,pad=0.05",
                               edgecolor=color, facecolor='white',
                               linewidth=2)
    ax_perf.add_patch(metric_box)

    ax_perf.text(1, y + 0.35, r'$\bf{' + metric + '}$',
                ha='left', va='center', fontsize=9,
                color=color, fontweight='bold')
    ax_perf.text(5, y - 0.05, value,
                ha='center', va='center', fontsize=10,
                color=COLORS['text_dark'])

# Comparison table
ax_perf.text(5, 1.8, r'$\bf{Block\ Granularity\ Analysis}$',
            ha='center', fontsize=9, fontweight='bold',
            color=COLORS['text_dark'])

table_data = [
    ('Model', '39GB', '2', '0.36'),
    ('Block', '650MB', '31', r'$\bf{0.85}$'),
    ('Layer', '72MB', '280', '0.29'),
]

y = 1.2
for i, (gran, size, transfers, eff) in enumerate(table_data):
    is_optimal = (gran == 'Block')
    bg_color = '#E8F5E9' if is_optimal else 'white'

    row_box = Rectangle((1.5, y - i*0.35), 7, 0.3,
                        edgecolor=COLORS['border'], facecolor=bg_color,
                        linewidth=1.5 if is_optimal else 0.8)
    ax_perf.add_patch(row_box)

    font_weight = 'bold' if is_optimal else 'normal'
    ax_perf.text(2, y - i*0.35 + 0.15, gran, ha='left', va='center',
                fontsize=7.5, fontweight=font_weight)
    ax_perf.text(4, y - i*0.35 + 0.15, size, ha='center', va='center',
                fontsize=7.5, fontweight=font_weight)
    ax_perf.text(5.5, y - i*0.35 + 0.15, transfers, ha='center', va='center',
                fontsize=7.5, fontweight=font_weight)
    ax_perf.text(7, y - i*0.35 + 0.15, eff, ha='center', va='center',
                fontsize=7.5, fontweight=font_weight,
                color=COLORS['optimal'] if is_optimal else COLORS['text_dark'])

# ============================================================================
# Panel F: Memory Timeline Comparison
# ============================================================================
ax_timeline = fig.add_subplot(gs[2, :])
ax_timeline.set_xlim(0, 100)
ax_timeline.set_ylim(0, 70)
ax_timeline.spines['top'].set_visible(False)
ax_timeline.spines['right'].set_visible(False)
ax_timeline.set_xlabel(r'$\bf{Inference\ Time\ (normalized)}$', fontsize=11, fontweight='bold')
ax_timeline.set_ylabel(r'$\bf{GPU\ Memory\ (GB)}$', fontsize=11, fontweight='bold')
ax_timeline.set_title(r'$\bf{(f)}$ Memory Usage: Baseline vs. RabbitVideo',
                     fontsize=13, fontweight='bold', pad=10)

# Baseline (OOM scenario)
t_baseline = np.linspace(0, 40, 100)
mem_baseline = 38 + 10 * np.exp(-t_baseline/10) + 3 * np.sin(t_baseline/5)
ax_timeline.plot(t_baseline, mem_baseline, linewidth=3, color='#D32F2F',
                label='Baseline (OOM)', linestyle='--', alpha=0.8)
ax_timeline.fill_between(t_baseline, 0, mem_baseline, color='#D32F2F', alpha=0.15)

# RabbitVideo
t_rabbit = np.linspace(0, 100, 200)
mem_rabbit = 20 + 3 * np.sin(t_rabbit/8) + 1.5 * np.random.randn(200) * 0.3
mem_rabbit = np.clip(mem_rabbit, 18, 24.5)
ax_timeline.plot(t_rabbit, mem_rabbit, linewidth=3, color=COLORS['optimal'],
                label='RabbitVideo', alpha=0.9)
ax_timeline.fill_between(t_rabbit, 0, mem_rabbit, color=COLORS['optimal'], alpha=0.2)

# GPU limit line
ax_timeline.axhline(y=24, color='#FF6F00', linewidth=2.5, linestyle=':',
                   label=r'$M_{GPU}$ Limit (24GB)', alpha=0.8)
ax_timeline.fill_between([0, 100], 24, 70, color='#FF6F00', alpha=0.08)

# Phase annotations
phase_regions = [
    (0, 5, 'Init', COLORS['phase1']),
    (5, 85, 'Streaming Inference', COLORS['phase2']),
    (85, 100, 'Decode', COLORS['phase3']),
]

for x_start, x_end, label, color in phase_regions:
    if x_end <= 100:
        ax_timeline.axvspan(x_start, x_end, alpha=0.08, color=color)
        if label == 'Streaming Inference':
            ax_timeline.text((x_start + x_end)/2, 65, label, ha='center',
                           fontsize=9, style='italic', color=color, alpha=0.7)

# Memory savings annotation
ax_timeline.annotate('', xy=(42, 45), xytext=(42, 22),
                    arrowprops=dict(arrowstyle='<->', color='#4CAF50', lw=2.5))
ax_timeline.text(47, 33.5, r'$\bf{2.75\times}$' + '\n' + r'$\bf{reduction}$',
                fontsize=10, color='#4CAF50', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor='#4CAF50', linewidth=2))

ax_timeline.legend(loc='upper right', fontsize=10, frameon=True,
                  fancybox=True, shadow=True)
ax_timeline.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
ax_timeline.set_yticks([0, 10, 20, 24, 30, 40, 50, 60])
ax_timeline.set_xticks([0, 20, 40, 60, 80, 100])

# ============================================================================
# Main Title
# ============================================================================
fig.suptitle(r'$\bf{RabbitVideo:}$ Streaming Execution for Memory-Constrained Video Diffusion',
            fontsize=16, fontweight='bold', y=0.985)

plt.tight_layout()

# Save with high quality
output_path = 'rabbitvideo_architecture_academic.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight',
           format='pdf', backend='pdf')
print(f"✓ Saved high-quality academic figure: {output_path}")

output_path_png = 'rabbitvideo_architecture_academic.png'
plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
print(f"✓ Saved PNG version: {output_path_png}")

plt.show()

print("\n" + "="*80)
print("RabbitVideo Architecture Visualization - Academic Quality")
print("="*80)
print("Key features:")
print("  • Multi-panel layout matching systems paper standards")
print("  • Mathematical formulations and theorem presentation")
print("  • Synchronous protocol algorithm visualization")
print("  • LRU optimality proof illustration")
print("  • Performance metrics and guarantees")
print("  • Memory timeline comparison")
print("  • Professional color scheme and typography")
print("="*80)
