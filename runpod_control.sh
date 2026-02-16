#!/bin/bash
# RunPod control script - start/stop pods to save costs
# Uses the new REST API (not GraphQL)

set -e

# Get API key from env or file
RUNPOD_API_KEY="${RUNPOD_API_KEY:-$(cat ~/.runpod_api_key 2>/dev/null)}"

if [ -z "$RUNPOD_API_KEY" ]; then
    echo "Error: RUNPOD_API_KEY not set"
    echo "Set it with: export RUNPOD_API_KEY=your_key"
    echo "Or save to ~/.runpod_api_key"
    exit 1
fi

API_BASE="https://rest.runpod.io/v1"

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

api_call() {
    local method="$1"
    local endpoint="$2"
    local data="$3"
    
    local response
    if [ -n "$data" ]; then
        response=$(curl -s -X "$method" "${API_BASE}${endpoint}" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $RUNPOD_API_KEY" \
            -d "$data")
    else
        response=$(curl -s -X "$method" "${API_BASE}${endpoint}" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $RUNPOD_API_KEY")
    fi
    
    # Check for errors
    if echo "$response" | jq -e '.error' >/dev/null 2>&1; then
        echo "API Error: $(echo "$response" | jq -r '.error')" >&2
        return 1
    fi
    
    echo "$response"
}

list_pods() {
    echo "Listing pods..."
    api_call GET "/pods" | jq '.[] | {id, name, status: .desiredStatus, gpu: .machine.gpu}'
}

get_pod_id() {
    local pods
    pods=$(api_call GET "/pods")
    echo "$pods" | jq -r ".[] | select(.name == \"$POD_NAME\") | .id" 2>/dev/null || echo ""
}

start_pod() {
    local POD_ID
    POD_ID=$(get_pod_id)
    
    if [ -n "$POD_ID" ] && [ "$POD_ID" != "null" ]; then
        echo "Starting existing pod: $POD_ID"
        api_call POST "/pods/$POD_ID/start" | jq
    else
        echo "Creating new pod: $POD_NAME"
        echo "GPU: $GPU_COUNT x $GPU_TYPE"
        echo "This will cost ~\$13-16/hr for 8x A100"
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
        
        # Create pod - use 'image' not 'imageName' per API response sample
        local payload
        payload=$(cat <<EOF
{
    "name": "$POD_NAME",
    "image": "$CONTAINER_IMAGE",
    "gpuTypeId": "NVIDIA A100 80GB PCIe",
    "gpuCount": $GPU_COUNT,
    "volumeInGb": $VOLUME_SIZE
}
EOF
)
        echo "Debug payload: $payload" >&2
        api_call POST "/pods" "$payload" | jq
    fi
}

stop_pod() {
    local POD_ID
    POD_ID=$(get_pod_id)
    if [ -z "$POD_ID" ] || [ "$POD_ID" == "null" ]; then
        echo "Pod not found: $POD_NAME"
        exit 1
    fi
    
    echo "Stopping pod: $POD_ID"
    api_call POST "/pods/$POD_ID/stop" | jq
}

delete_pod() {
    local POD_ID
    POD_ID=$(get_pod_id)
    if [ -z "$POD_ID" ] || [ "$POD_ID" == "null" ]; then
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
    api_call DELETE "/pods/$POD_ID" | jq
}

get_status() {
    local POD_ID
    POD_ID=$(get_pod_id)
    if [ -z "$POD_ID" ] || [ "$POD_ID" == "null" ]; then
        echo "Pod not found: $POD_NAME"
        echo "Run '$0 start' to create it"
        exit 1
    fi
    
    echo "Pod Status:"
    api_call GET "/pods/$POD_ID" | jq '{
        id: .id,
        name: .name,
        status: .desiredStatus,
        gpu: .machine.gpu,
        gpuCount: .machine.gpuCount,
        publicIp: .runtime.publicIp,
        ports: .runtime.ports
    }'
}

ssh_pod() {
    local POD_ID
    POD_ID=$(get_pod_id)
    if [ -z "$POD_ID" ] || [ "$POD_ID" == "null" ]; then
        echo "Pod not found: $POD_NAME"
        exit 1
    fi
    
    # Get pod info
    local pod_info
    pod_info=$(api_call GET "/pods/$POD_ID")
    
    local SSH_IP SSH_PORT
    SSH_IP=$(echo "$pod_info" | jq -r '.runtime.publicIp // empty')
    SSH_PORT=$(echo "$pod_info" | jq -r '.runtime.ports[] | select(.privatePort == 22) | .publicPort // empty')
    
    if [ -z "$SSH_IP" ]; then
        echo "SSH not available. Pod may be starting..."
        echo "Current status: $(echo "$pod_info" | jq -r '.desiredStatus')"
        exit 1
    fi
    
    echo "Connecting to: root@$SSH_IP:$SSH_PORT"
    ssh -o StrictHostKeyChecking=no -p "$SSH_PORT" "root@$SSH_IP"
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
