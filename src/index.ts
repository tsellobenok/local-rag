import { performance } from 'perf_hooks';
import {
  Document,
  HuggingFaceEmbedding,
  MetadataMode,
  NodeWithScore,
  Ollama,
  Settings,
  storageContextFromDefaults,
  VectorStoreIndex,
} from 'llamaindex';
import { data } from './data';

Settings.llm = new Ollama({ model: 'phi4' });

Settings.embedModel = new HuggingFaceEmbedding({
  modelType: 'Snowflake/snowflake-arctic-embed-l-v2.0',
  modelOptions: { dtype: 'fp16' },
});

const embeddingStart = performance.now();
const storageContext = await storageContextFromDefaults({
  persistDir: './storage',
});

const index = await VectorStoreIndex.fromDocuments(
  data.map(({ text, id }) => new Document({ text, id_: id.toString() })),
  { storageContext },
);

process.stdout.write(`Embedding time: ${Math.round((performance.now() - embeddingStart) / 1000)}s\n`);
process.stdout.write('---------------------------------------------------------------\n\n');

const queryStart = performance.now();
const queryEngine = index.asQueryEngine();
const stream = await queryEngine.query({
  query: 'What is the answer to the question of nothing? Give me just the answer, no additional comments or notes.',
  stream: true,
});

const sourceNodes: NodeWithScore[] = [];

for await (const chunk of stream) {
  process.stdout.write(chunk.message.content as string);

  if (chunk.sourceNodes && !sourceNodes.length) {
    sourceNodes.push(...chunk.sourceNodes);
  }
}

process.stdout.write('\n\n');
process.stdout.write(`Query time: ${Math.round((performance.now() - queryStart) / 1000)}s\n`);
process.stdout.write('---------------------------------------------------------------\n\n');

if (sourceNodes) {
  sourceNodes.forEach((source) => {
    console.log(`Score: ${source.score} - ${source.node.getContent(MetadataMode.NONE).slice(0, 100)}...\n`);
  });
}
