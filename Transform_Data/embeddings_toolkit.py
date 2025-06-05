import logging

import numpy as np
from sentence_transformers import SentenceTransformer

# logging, Warning level, to file
logging.basicConfig(level=logging.DEBUG)

sentences = ["The device: PDX-RO details\n Hostname: PDX-RO\n Location: Global/OR/PDX/Floor-2\n Device Role: BORDER ROUTER"]

print('\n\n' +sentences[0] + '\n\n')

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)[0]

print(embeddings)

# Save to a text file
np.savetxt("output.txt", embeddings)
