# Catalyst Center Retrieval-Augmented Generation GenAI for Ops Data


The repo includes the files to:

- Create and run a Chrom DB vector database server.
It will create the folder to store the data and start the server.
A second app will allow to erase the vector database and or create a new vector database.

- Create Embeddings
All the packages to create embeddings and save them to local or server vector database.

- Similarity searches using GTP-3.5-turbo, GPT-4o and Ollama/Llama3 models
The folder includes the files for a client to query a ChromaDB database using few different LLM models

Sample Output:

GPT-3.5-turbo:
```shell

How may I help you?  Who are the CDP neighbors?
The CDP neighbors listed in the output are:
1. Device ID: LO-BN with IP address 10.93.140.7, platform Cisco C9300-24U, and connected to the local interface TenGigabitEthernet1/0/23.
2. Device ID: PDX-M with IP address 10.93.141.1, platform Cisco WS-C3850-48U, and connected to the local interface GigabitEthernet0/0.
3. Device ID: PDX-M with IP address 10.93.141.33, platform Cisco WS-C3850-48U, and connected to the local interface TenGigabitEthernet1/0/24.


How may I help you?  What are the configured routing protocols?
The configured routing protocols on the device are EIGRP (Autonomous System 123), LISP, and BGP (Autonomous System 65100).

```

GPT-40:
```shell

How may I help you?  Who are the CDP neighbors?
The CDP neighbors are:

1. **LO-BN**
   - Local Interface: TenGigabitEthernet1/0/23
   - Holdtime: 167 seconds
   - Platform: cisco C9300-24U
   - Port ID: GigabitEthernet1/0/23
   - IP Address: 10.93.140.7

2. **PDX-M** (Two entries with the same device ID but different port details)
   - Local Interface: GigabitEthernet0/0
   - Holdtime: 143 seconds
   - Platform: cisco WS-C3850-48U
   - Port ID: GigabitEthernet1/0/4
   - IP Address: 10.93.141.1

   - Local Interface: TenGigabitEthernet1/0/24
   - Holdtime: 148 seconds
   - Platform: cisco WS-C3850-48U
   - Port ID: GigabitEthernet1/0/24
   - IP Address: 10.93.141.33


How may I help you?  What are the configured routing protocols?
The configured routing protocols are:

1. EIGRP (Enhanced Interior Gateway Routing Protocol) with Autonomous System number 123.
2. LISP (Locator/ID Separation Protocol).
3. BGP (Border Gateway Protocol) with Autonomous System number 65100.

```

Ollama/Llama3:
```shell

How may I help you?  Who are the CDP neighbors?
Based on the output of the "show cdp neighbors" command, the CDP (Cisco Discovery Protocol) neighbors are:

* Device ID: LO-BN, Local Intrfce: TenGigabitEthernet1/0/23
* Device ID: PDX-M, Local Intrfce: Gig 0/0 or TenGigabitEthernet1/0/24

These two devices are the CDP neighbors of the device running the command.


How may I help you?  What are the configured routing protocols?
Based on the given context, the configured routing protocols are:

1. EIGRP (Extended Interior Gateway Routing Protocol) with AS number 123.
2. BGP (Border Gateway Protocol) with autonomous system number 65100.
3. LISP (Label Switched Path) is also mentioned as a routing protocol.

These protocols are listed in the output of the command "show ip protocols".

```

ChromaDB Server logging:
```shell
/Users/gzapodea/PythonCode/catalyst_center_rag_ops


                (((((((((    (((((####
             ((((((((((((((((((((((#########
           ((((((((((((((((((((((((###########
         ((((((((((((((((((((((((((############
        (((((((((((((((((((((((((((#############
        (((((((((((((((((((((((((((#############
         (((((((((((((((((((((((((##############
         ((((((((((((((((((((((((##############
           (((((((((((((((((((((#############
             ((((((((((((((((##############
                (((((((((    #########

    

Running Chroma

Saving data to: VDB
Connect to chroma at: http://localhost:8010
Getting started guide: https://docs.trychroma.com/getting-started


INFO:     [31-05-2024 18:10:19] Set chroma_server_nofile to 65535
INFO:     [31-05-2024 18:10:19] Anonymized telemetry enabled. See                     https://docs.trychroma.com/telemetry for more information.
DEBUG:    [31-05-2024 18:10:19] Starting component System
DEBUG:    [31-05-2024 18:10:19] Starting component OpenTelemetryClient
DEBUG:    [31-05-2024 18:10:19] Starting component SimpleAssignmentPolicy
DEBUG:    [31-05-2024 18:10:19] Starting component SqliteDB
DEBUG:    [31-05-2024 18:10:19] Starting component QuotaEnforcer
DEBUG:    [31-05-2024 18:10:19] Starting component Posthog
DEBUG:    [31-05-2024 18:10:19] Starting component LocalSegmentManager
DEBUG:    [31-05-2024 18:10:19] Starting component SegmentAPI
INFO:     [31-05-2024 18:10:19] Started server process [6236]
INFO:     [31-05-2024 18:10:19] Waiting for application startup.
INFO:     [31-05-2024 18:10:19] Application startup complete.
INFO:     [31-05-2024 18:10:19] Uvicorn running on http://localhost:8010 (Press CTRL+C to quit)
INFO:     [31-05-2024 18:10:31] ::1:51209 - "GET /api/v1/tenants/default_tenant HTTP/1.1" 200
INFO:     [31-05-2024 18:10:31] ::1:51209 - "GET /api/v1/databases/default_database?tenant=default_tenant HTTP/1.1" 200
INFO:     [31-05-2024 18:10:36] ::1:51213 - "POST /api/v1/collections?tenant=default_tenant&database=default_database HTTP/1.1" 200
DEBUG:    [31-05-2024 18:10:46] Starting component PersistentLocalHnswSegment
INFO:     [31-05-2024 18:10:46] ::1:51213 - "POST /api/v1/collections/d58709ae-edb6-4ad3-a261-412355c4291a/query HTTP/1.1" 200
INFO:     [31-05-2024 18:10:52] ::1:51222 - "GET /api/v1/tenants/default_tenant HTTP/1.1" 200
INFO:     [31-05-2024 18:10:52] ::1:51222 - "GET /api/v1/databases/default_database?tenant=default_tenant HTTP/1.1" 200
INFO:     [31-05-2024 18:10:56] ::1:51226 - "POST /api/v1/collections?tenant=default_tenant&database=default_database HTTP/1.1" 200
INFO:     [31-05-2024 18:11:25] ::1:51236 - "POST /api/v1/collections/d58709ae-edb6-4ad3-a261-412355c4291a/query HTTP/1.1" 200
INFO:     [31-05-2024 18:11:34] ::1:51239 - "POST /api/v1/collections/d58709ae-edb6-4ad3-a261-412355c4291a/query HTTP/1.1" 200
INFO:     [31-05-2024 18:11:44] ::1:51239 - "POST /api/v1/collections/d58709ae-edb6-4ad3-a261-412355c4291a/query HTTP/1.1" 200
INFO:     [31-05-2024 18:14:02] ::1:51284 - "GET /api/v1/tenants/default_tenant HTTP/1.1" 200
INFO:     [31-05-2024 18:14:02] ::1:51284 - "GET /api/v1/databases/default_database?tenant=default_tenant HTTP/1.1" 200
INFO:     [31-05-2024 18:14:06] ::1:51289 - "POST /api/v1/collections?tenant=default_tenant&database=default_database HTTP/1.1" 200
INFO:     [31-05-2024 18:14:17] ::1:51289 - "POST /api/v1/collections/d58709ae-edb6-4ad3-a261-412355c4291a/query HTTP/1.1" 200
INFO:     [31-05-2024 18:15:53] ::1:51346 - "POST /api/v1/collections/d58709ae-edb6-4ad3-a261-412355c4291a/query HTTP/1.1" 200

```
