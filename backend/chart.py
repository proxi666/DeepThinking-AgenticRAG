import matplotlib.pyplot as plt

metrics = ['Faithfulness', 'Context\nRecall', 'Context\nPrecision', 'Answer\nCorrectness']
baseline = [0.345, 0.117, 0.0, 0.491]
deep_thinking = [0.534, 0.150, 0.299, 0.500]

x = range(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar([i - width/2 for i in x], baseline, width, label='Baseline RAG', alpha=0.8)
plt.bar([i + width/2 for i in x], deep_thinking, width, label='Deep Thinking RAG', alpha=0.8)

plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('RAG System Comparison')
plt.xticks(x, metrics)
plt.legend()
plt.ylim(0, 0.6)
plt.grid(axis='y', alpha=0.3)

plt.savefig('evaluation_comparison.png', dpi=300, bbox_inches='tight')
plt.show()