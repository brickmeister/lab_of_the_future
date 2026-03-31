import React from 'react'
import ReactDOM from 'react-dom/client'
import { AppKitProvider } from '@databricks/appkit/react'
import App from './App'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <AppKitProvider>
      <App />
    </AppKitProvider>
  </React.StrictMode>,
)
