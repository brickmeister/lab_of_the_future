import { createServer } from '@databricks/appkit/server';

const server = createServer({
  queriesPath: './config/queries',
});

server.start();
