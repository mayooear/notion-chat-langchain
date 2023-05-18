import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OpenAIEmbeddings } from 'langchain/embeddings';
import { PineconeStore } from 'langchain/vectorstores';
import { pinecone } from '@/utils/pinecone-client';
import { processMarkDownFiles } from '@/utils/helpers';
import { PINECONE_INDEX_NAME, PINECONE_NAME_SPACE } from '@/config/pinecone';

/* Name of directory to retrieve files from. You can change this as required */
const directoryPath = 'Notion_DB';

async function ingestData(index: string, docs: any[], embeddings: any[], chunkSize: number) {
  for (let i = 0; i < docs.length; i += chunkSize) {
    const chunk = docs.slice(i, i + chunkSize);
    try {
      await PineconeStore.fromDocuments(
        index,
        chunk,
        embeddings,
        'text',
        PINECONE_NAME_SPACE, // optional namespace for your vectors
      );
      console.log(`Successfully ingested chunk ${i / chunkSize + 1}`);
    } catch (error) {
      console.error(`Error ingesting chunk ${i / chunkSize + 1}:`, error);
      // Handle the error as needed
      throw new Error('Failed to ingest your data');
    }
  }
}

export const run = async () => {
  try {
    /* Load raw docs from the markdown files in the directory */
    const rawDocs = await processMarkDownFiles(directoryPath);

    /* Split text into chunks */
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const docs = await textSplitter.splitDocuments(rawDocs);
    console.log('split docs', docs);

    console.log('creating vector store...');
    /* Create and store the embeddings in the vectorStore */
    const embeddings = new OpenAIEmbeddings();
    const index = pinecone.Index(PINECONE_INDEX_NAME); // change to your own index name

    await ingestData(index, docs, embeddings, 1000);

  } catch (error) {
    console.log('error', error);
    throw new Error('Failed to ingest your data');
  }
};

(async () => {
  await run();
  console.log('ingestion complete');
})();
