import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OpenAIEmbeddings } from 'langchain/embeddings';
import { PineconeStore } from 'langchain/vectorstores';
import { PineconeLibArgs } from 'langchain/vectorstores/pinecone.js';
import { pinecone } from '@/utils/pinecone-client';
import { processMarkDownFiles } from '@/utils/helpers';
import { PINECONE_INDEX_NAME, PINECONE_NAME_SPACE } from '@/config/pinecone';
import type { VectorOperationsApi } from "@pinecone-database/pinecone/dist/pinecone-generated-ts-fetch";
/* Name of directory to retrieve files from. You can change this as required */
const directoryPath = 'Notion_DB';
interface PineconeLibArgs {
    pineconeIndex: VectorOperationsApi;
    textKey?: string;
    namespace?: string;
}

export const run = async () => {
  try {
    /*load raw docs from the markdown files in the directory */
    const rawDocs = await processMarkDownFiles(directoryPath);

    /* Split text into chunks */
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const docs = await textSplitter.splitDocuments(rawDocs);
    // console.log('split docs', docs);
    console.log('split docs ok')
    console.log('docs length', docs.length)
    for (let i = 0; i < docs.length; i += 100) {
        const slice = docs.slice(i, i + 100);
        console.log('creating vector store...');
        /*create and store the embeddings in the vectorStore*/
        const embeddings = new OpenAIEmbeddings();
        console.log('initiated embeddings');
        const index = pinecone.Index(PINECONE_INDEX_NAME); //change to your own index name
        console.log('initiated index');
        let dbConfig:PineconeLibArgs = {
            pineconeIndex: pinecone.Index(PINECONE_INDEX_NAME),
            namespace: PINECONE_NAME_SPACE,
        };
        await PineconeStore.fromDocuments(
          slice,
          embeddings,
          dbConfig
        );
        // Do something with the slice
      }


  } catch (error) {
    console.log('error', error);
    throw new Error('Failed to ingest your data');
  }
};

(async () => {
  await run();
  console.log('ingestion complete');
})();
