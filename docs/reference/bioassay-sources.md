# Bioassay data sources

ToxAgent's `get_pubchem_bioassays` resolves a submitted structure to a single
PubChem CID and reads the PubChem BioAssay summary for human hERG, matching
NCBI GeneID 3757 or UniProt Q12809. Its evidence records retain AID, SID, CID,
source link, outcome, assay type, activity name and reported value where
available. It honours the case's external-search restriction before sending
any structure. Only the first requested records are returned, and the total
matching count and truncation are explicit.

PubChem standardizes structures. Check stereochemistry, salts and the assay
protocol before relying on a match; screening results and active/inactive
labels do not establish safety or clinical risk. A PubChem assay and a ChEMBL
record can derive from the same publication or experiment and must not be
counted as independent evidence. Missing records do not establish inactivity.
The summary need not include a measurement qualifier or full protocol; check
the linked assay and original publication for quantitative interpretation.

On 2026-11-13, from the production control container, a single ChEMBL
structure lookup for caffeine returned HTTP 500 after 5.46 seconds. BindingDB's
compound endpoint returned 200 in 0.92 seconds. PubChem assay summaries for
caffeine (CID 2519) and terfenadine (CID 5405) returned 200 in 2.17 and 2.24
seconds respectively. These are point-in-time observations, not uptime or
latency guarantees. The caffeine summary was about 1.75 MB, so the assay
adapter bounds decoded responses at 12 MiB and makes one attempt per request.
It spaces its own requests to two per second per process; multi-process
production needs a shared limiter and must account for other PubChem clients.

- [PubChem PUG REST tutorial](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest-tutorial):
  structure resolution, assay summaries, field definitions, and usage policy.
- [BindingDB REST services](https://www.bindingdb.org/rwd/bind/BindingDBRESTfulAPI.jsp):
  compound similarity and target binding measurements. Similarity results
  require independent exact structure and target checks before use as evidence.
- [Microsoft Discovery BindingDB agent](https://github.com/microsoft/discovery/blob/main/agents/bindingdb/README.md):
  a local SQLite approach with a substantial snapshot and build footprint.
- [ChEMBL web services](https://github.com/chembl/GLaDOS-docs/blob/master/web-services/chembl-data-web-services.md):
  measurements, targets, and structure search. A different MCP wrapper over
  this same API does not remove its upstream availability dependency.

ChEMBL is no longer queried or exposed as a live tool. Its historical evidence
records remain readable, and PubChem may contain records contributed by other
databases. The MCP tool registry exposes PubChem BioAssay to decision-support
runs when the scientific-primitives flag and PubChem provider are enabled.
