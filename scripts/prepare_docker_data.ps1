# Run from any location: .\scripts\prepare_docker_data.ps1
# Creates a NEW copy for Docker; never writes to data/vectorstore.
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$sourceRoot = Join-Path $repoRoot 'data'
$destinationRoot = Join-Path $sourceRoot 'docker_runtime'
$sourceVector = Join-Path $sourceRoot 'vectorstore'
$destinationVector = Join-Path $destinationRoot 'vectorstore'

if (-not (Test-Path -LiteralPath $sourceVector -PathType Container)) {
    throw "Original vector store not found: $sourceVector"
}
if (Test-Path -LiteralPath $destinationRoot) {
    throw "Destination already exists; refusing to overwrite: $destinationRoot"
}
foreach ($name in @('knowledge_graph.graphml','knowledge_graph_metadata.json')) {
    if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot $name) -PathType Leaf)) {
        throw "Required graph file not found: $name"
    }
}

Write-Host 'Copying the database for Docker. Stop the local FastAPI service first.' -ForegroundColor Cyan
New-Item -ItemType Directory -Path $destinationVector -Force | Out-Null
foreach ($name in @('knowledge_graph.graphml','knowledge_graph_metadata.json')) {
    Copy-Item -LiteralPath (Join-Path $sourceRoot $name) -Destination (Join-Path $destinationRoot $name)
}

& robocopy $sourceVector $destinationVector /E /R:2 /W:2 /J /NP /NFL /NDL /NJH /NJS
$exit = $LASTEXITCODE
if ($exit -ge 8) {
    throw "robocopy reported an error ($exit). The Docker data copy may be incomplete."
}

$files = @(Get-ChildItem -LiteralPath $destinationVector -Recurse -File)
$total = ($files | Measure-Object -Property Length -Sum).Sum
Write-Host "Docker vector store copy: $($files.Count) files / $([math]::Round($total / 1GB, 2)) GB" -ForegroundColor Green
Write-Host "Copied into: $destinationRoot"
Write-Host 'The original vector store has not been modified by this script.'
