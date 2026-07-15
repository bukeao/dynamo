nohup etcd --name node1  --initial-cluster "node1=http://0.0.0.0:2480" --listen-peer-urls http://0.0.0.0:2480 --initial-advertise-peer-urls http://0.0.0.0:2480 --listen-client-urls http://0.0.0.0:2479 --advertise-client-urls http://0.0.0.0:2479 --data-dir /tmp/etcd &

nohup nats-server -js -p 5222 &

ulimit -n 65536

# VLLM_KV_CACHE_LAYOUT=HND VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_TARGET_DEVICE=xpu KVBM_DISABLE_CUSTOM_TRANSFER=0 KVBM_ALLOW_CUSTOM_TRANSFER_ON_ZE=1 KVBM_BATCH_TRANSFERS=1 

ZE_AFFINITY_MASK=2,3 NATS_SERVER="nats://127.0.0.1:5222" ETCD_ENDPOINTS="http://127.0.0.1:2479" DYN_KVBM_CPU_CACHE_GB="256"  DYN_VLLM_KV_EVENT_PORT="20040" VLLM_SERVER_DEV_MODE="1" DYN_LOG="info" DYN_KVBM_METRICS="true" FLASHINFER_DISABLE_VERSION_CHECK="1" RUST_BACKTRACE="1" python -m dynamo.vllm   --model openai/gpt-oss-20b --kv-transfer-config '{"kv_connector": "DynamoConnector", "kv_connector_module_path": "kvbm.vllm_integration.connector", "kv_role":"kv_both"}'   --discovery-backend file   --enforce-eager  --gpu-memory-utilization 0.9   --block-size 64 --tensor-parallel-size 2 > kvbm.log 2>&1
