import { Client } from '@web3-storage/w3up-client'
import { filesFromPaths } from 'files-from-path'
import * as Signer from '@ucanto/principal/ed25519'
import { Agent } from '@web3-storage/access'
import { StoreConf } from '@web3-storage/access/stores/store-conf'

const command = process.argv[2]

async function main() {
  switch(command) {
    case 'upload': {
      const [filePath, key] = process.argv.slice(3)
      const files = await filesFromPaths([filePath])
      
      // 1. Initialize agent with persistent storage
      const principal = Signer.parse(key)
      const agent = await Agent.create({
        principal,
        store: new StoreConf('storacha-agent')
      })

      // 2. Provision agent with service
      const provisioning = await agent.provision(principal)
      if (!provisioning.ok) {
        throw new Error('Failed to provision agent: ' + provisioning.error.message)
      }
      
      // 3. Initialize client with provisioned agent
      const client = new Client(agent)

      // 4. Create and register space
      let space = client.currentSpace()
      if (!space) {
        const spaceCreation = await client.createSpace('my-space', {
          provider: agent.did()
        })
        if (!spaceCreation.ok) throw new Error('Space creation failed')
        space = spaceCreation.ok
        await client.setCurrentSpace(space.did())
      }

      // 5. Upload file
      const uploadResult = await client.uploadFile(files[0])
      if (!uploadResult.ok) throw new Error('Upload failed: ' + uploadResult.error.message)
      
      console.log(JSON.stringify({
        cid: uploadResult.ok.toString(),
        filename: files[0].name,
        space: space.did()
      }))
      break
    }
    
    case 'download': {
      const [cid] = process.argv.slice(3)
      console.log(JSON.stringify({
        gateway_url: `https://w3s.link/ipfs/${cid}`
      }))
      break
    }
  }
}

main().catch(err => {
  console.error(err)
  process.exit(1)
})
