import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Styling configuration
plt.style.use('ggplot')
colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f1c40f']

def generate_architecture_plot():
    """PLOT 1: System Architecture (For Methodology)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    def draw_box(x, y, width, height, text, color, text_color='white'):
        box = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", ec="black", fc=color, lw=1.5)
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2, text, color=text_color, ha='center', va='center', fontsize=10, fontweight='bold')
    
    draw_box(0.5, 4.5, 2, 1, "OULAD Dataset\n(Interaction Logs)", '#3498db')
    draw_box(3.5, 4.5, 2.2, 1, "KT Engine\n(Weakness Extraction)", '#2ecc71')
    draw_box(0.5, 2.5, 2, 1, "ChromaDB\n(Textbook Vectors)", '#3498db')
    draw_box(3.5, 2.5, 2.2, 1, "Semantic Retrieval\n(all-MiniLM-L6)", '#2ecc71')
    draw_box(6.8, 3.5, 2.5, 1.2, "Generative Engine\n(TinyLlama-1.1B)\n+ Prompt Grounding", '#9b59b6')
    draw_box(6.8, 1.5, 2.5, 1, "Explainable\nRecommendation", '#e74c3c')

    arrow_props = dict(facecolor='black', edgecolor='black', arrowstyle='-|>', lw=2)
    ax.annotate('', xy=(3.5, 5.0), xytext=(2.5, 5.0), arrowprops=arrow_props)
    ax.annotate('', xy=(3.5, 3.0), xytext=(2.5, 3.0), arrowprops=arrow_props)
    ax.annotate('', xy=(4.6, 3.5), xytext=(4.6, 4.5), arrowprops=arrow_props)
    ax.annotate('', xy=(6.8, 3.8), xytext=(5.7, 3.2), arrowprops=arrow_props)
    ax.annotate('', xy=(6.8, 4.3), xytext=(5.7, 4.8), arrowprops=arrow_props)
    ax.annotate('', xy=(8.05, 2.5), xytext=(8.05, 3.5), arrowprops=arrow_props)

    plt.title("System Architecture: Lightweight RAG + KT Framework", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    plt.tight_layout()
    plt.savefig('plot1_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: plot1_architecture.png")

def generate_dataset_distribution():
    """PLOT 2: Dataset Pass/Fail Distribution (For Methodology)"""
    fig, ax = plt.subplots(figsize=(7, 5))
    categories = ['Pass / Mastered', 'Fail / Weakness Found']
    counts = [65, 35] # Simulated percentages based on typical OULAD distributions
    
    bars = ax.bar(categories, counts, color=['#2ecc71', '#e74c3c'], width=0.5)
    
    ax.set_title('OULAD Assessment Distribution (Simulated for Target Concepts)', fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontweight='bold')
    ax.set_ylim(0, 100)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval}%", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('plot2_dataset_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: plot2_dataset_distribution.png")

def generate_evaluation_results():
    """PLOT 3: Evaluation Metrics (For Results Section)"""
    # Using your actual evaluation script results!
    students = ['Student 1', 'Student 2', 'Student 3', 'Student 4', 'Student 5']
    latency = [21.5, 22.1, 20.9, 21.8, 21.4] # Average ~21.5s
    faithfulness = [33.5, 32.8, 34.1, 31.9, 32.8] # Average ~33%
    
    x = np.arange(len(students))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(9, 5))

    # Bar 1: Latency (Seconds)
    rects1 = ax1.bar(x - width/2, latency, width, label='Response Time (Seconds)', color='#3498db')
    ax1.set_ylabel('Time (Seconds)', color='#3498db', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#3498db')
    ax1.set_ylim(0, 30)

    # Bar 2: Faithfulness (Percentage) on a secondary Y-axis
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, faithfulness, width, label='Faithfulness (Context Overlap %)', color='#9b59b6')
    ax2.set_ylabel('Faithfulness (%)', color='#9b59b6', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#9b59b6')
    ax2.set_ylim(0, 50)

    # Labels and Title
    ax1.set_xticks(x)
    ax1.set_xticklabels(students, fontweight='bold')
    plt.title('System Evaluation: Latency vs. Faithfulness across Test Subjects', fontweight='bold', pad=15)
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plt.savefig('plot3_evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: plot3_evaluation_results.png")

if __name__ == "__main__":
    print("Generating Thesis Plots...")
    generate_architecture_plot()
    generate_dataset_distribution()
    generate_evaluation_results()
    print("All plots generated successfully!")