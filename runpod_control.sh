#!/bin/bash
# RunPod control script - start/stop pods to save costs

set -e

# Get API key from env or file
RUNPOD_API_KEY="${RUNPOD_API_KEY:-$(cat ~/.runpod_api_key 2>/dev/null)}"

if [ -z "$RUNPOD_API_KEY" ]; then
    echo "Error: RUNPOD_API_KEY not set"
    echo "Set it with: export RUNPOD_API_KEY=your_key"
    echo "Or save to ~/.runpod_api_key"
    exit 1
fi

API_URL="https://api.runpod.io/graphql"

# Pod configuration
POD_NAME="${POD_NAME:-world-model-inference}"
GPU_TYPE="${GPU_TYPE:-NVIDIA A100 80GB PCIe}"
GPU_COUNT="${GPU_COUNT:-8}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
VOLUME_SIZE="${VOLUME_SIZE:-100}"

usage() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  list     - List all pods"
    echo "  start    - Start/create the world model pod"
    echo "  stop     - Stop the pod (keeps storage)"
    echo "  delete   - Delete the pod completely"
    echo "  status   - Show pod status and connection info"
    echo "  ssh      - SSH into the pod"
}

graphql_query() {
    local query="$1"
    local response
    response=$(curl -s -X POST "$API_URL" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -d "{\"query\": \"$query\"}")
    
    # Debug: show raw response if it's not valid JSON
    if ! echo "$response" | jq . >/dev/null 2>&1; then
        echo "API Error (raw response): $response" >&2
        echo "{}"
        return 1
    fi
    echo "$response"
}

list_pods() {
    echo "Listing pods..."
    graphql_query "{ myself { pods { id name runtime { uptimeInSeconds gpus { id } } } } }" | jq '.data.myself.pods'
}

get_pod_id() {
    local response
    response=$(graphql_query "{ myself { pods { id name } } }")
    echo "$response" | jq -r ".data.myself.pods[] | select(.name == \"$POD_NAME\") | .id" 2>/dev/null || echo ""
}

start_pod() {
    POD_ID=$(get_pod_id)
    
    if [ -n "$POD_ID" ]; then
        echo "Starting existing pod: $POD_ID"
        graphql_query "mutation { podResume(input: { podId: \"$POD_ID\" }) { id } }" | jq
    else
        echo "Creating new pod: $POD_NAME"
        echo "GPU: $GPU_COUNT x $GPU_TYPE"
        echo "This will cost ~\$13-16/hr for 8x A100"
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
        
        # Create pod mutation
        MUTATION="mutation {
            podFindAndDeployOnDemand(input: {
                name: \\\"$POD_NAME\\\",
                imageName: \\\"$CONTAINER_IMAGE\\\",
                gpuTypeId: \\\"NVIDIA A100 80GB PCIe\\\",
                gpuCount: $GPU_COUNT,
                volumeInGb: $VOLUME_SIZE,
                containerDiskInGb: 50,
                minMemoryInGb: 100,
                minVcpuCount: 16,
                ports: \\\"22/tcp,8765/tcp\\\",
                startSsh: true
            }) {
                id
                machineId
            }
        }"
        
        graphql_query "$MUTATION" | jq
    fi
}

stop_pod() {
    POD_ID=$(get_pod_id)
    if [ -z "$POD_ID" ]; then
        echo "Pod not found: $POD_NAME"
        exit 1
    fi
    
    echo "Stopping pod: $POD_ID"
    graphql_query "mutation { podStop(input: { podId: \"$POD_ID\" }) { id } }" | jq
}

delete_pod() {
    POD_ID=$(get_pod_id)
    if [ -z "$POD_ID" ]; then
        echo "Pod not found: $POD_NAME"
        exit 1
    fi
    
    echo "WARNING: This will delete the pod and ALL DATA"
    read -p "Are you sure? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    
    echo "Deleting pod: $POD_ID"
    graphql_query "mutation { podTerminate(input: { podId: \"$POD_ID\" }) }" | jq
}

get_status() {
    POD_ID=$(get_pod_id)
    if [ -z "$POD_ID" ]; then
        echo "Pod not found: $POD_NAME"
        echo "Run '$0 start' to create it"
        exit 1
    fi
    
    echo "Pod Status:"
    graphql_query "{ pod(input: { podId: \"$POD_ID\" }) { 
        id name 
        desiredStatus
        runtime { 
            uptimeInSeconds 
            ports { ip isIpPublic publicPort privatePort type }
            gpus { id gpuUtilPercent memoryUtilPercent }
        }
    }}" | jq '.data.pod'
}

ssh_pod() {
    POD_ID=$(get_pod_id)
    if [ -z "$POD_ID" ]; then
        echo "Pod not found: $POD_NAME"
        exit 1
    fi
    
    # Get SSH info
    SSH_INFO=$(graphql_query "{ pod(input: { podId: \"$POD_ID\" }) { 
        runtime { ports { ip publicPort privatePort type } }
    }}")
    
    SSH_IP=$(echo "$SSH_INFO" | jq -r '.data.pod.runtime.ports[] | select(.privatePort == 22) | .ip')
    SSH_PORT=$(echo "$SSH_INFO" | jq -r '.data.pod.runtime.ports[] | select(.privatePort == 22) | .publicPort')
    
    if [ -z "$SSH_IP" ] || [ "$SSH_IP" == "null" ]; then
        echo "SSH not available. Pod may be starting..."
        exit 1
    fi
    
    echo "Connecting to: root@$SSH_IP:$SSH_PORT"
    ssh -p "$SSH_PORT" "root@$SSH_IP"
}

case "${1:-}" in
    list)   list_pods ;;
    start)  start_pod ;;
    stop)   stop_pod ;;
    delete) delete_pod ;;
    status) get_status ;;
    ssh)    ssh_pod ;;
    *)      usage ;;
esac
