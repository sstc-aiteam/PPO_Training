from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
import subprocess
import os
import json
import zipfile
import shutil
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
import logging
from pathlib import Path
import uuid
from typing import Optional, Dict, Any, List, Union
import re
import json
from datetime import datetime

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PPO Training API with Data Upload", version="2.0.0")

# 全局變量存儲訓練狀態
training_status = {
    "is_running": False,
    "current_job": None,
    "last_job": None,
    "logs": []
}

class TrainingConfig(BaseModel):
    work_dir: str = "/home/itrib30156/llm_vision/LLaMA-Factory/LLaMA-Factory"
    parent_dir: str = "/home/itrib30156/llm_vision/LLaMA-Factory"
    data_dir: str = "/home/itrib30156/llm_vision/LLaMA-Factory/data/ppo_data"
    base_model: str = "/home/itrib30156/llm_vision/phi4"
    # adapter_path: str = "/home/itrib30156/llm_vision/LLaMA-Factory/ppo_model/checkpoint-650"
    reward_model: str = "/home/itrib30156/llm_vision/LLaMA-Factory/rm_model/checkpoint-600"
    ref_model: str = "/home/itrib30156/llm_vision/phi4"
    
    # 訓練參數 - 使用你的默認值
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    learning_rate: float = 0.000005
    num_train_epochs: int = 1
    lora_rank: int = 64
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    max_length: int = 512
    max_new_tokens: int = 128
    temperature: float = 0.7
    seed: int = 42
    
    # GPU 設置 - 使用你的默認值
    gpu_devices: str = "0,1"
    nproc_per_node: int = 2
    
    # 數據集名稱
    dataset_name: str = "ppo_data"
    
    # 額外的訓練參數（可選）
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    logging_steps: int = 5
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    dataloader_num_workers: int = 4
    lr_scheduler_type: str = "cosine"

# Fixed DataUploadResponse model
class DataUploadResponse(BaseModel):
    message: str
    dataset_id: str
    dataset_name: str
    dataset_info: Dict[str, Any]
    files_processed: Dict[str, Union[str, int]]  # Changed to allow both str and int values
    upload_time: str

class TrainingJob:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.run_name = f"qwen_ppo_lora_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.process = None
        self.start_time = None
        self.end_time = None
        self.status = "pending"
        self.logs = []
        
    def validate_paths(self):
        """驗證所需的路徑和文件是否存在"""
        required_files = [
            f"{self.config.data_dir}/dataset_info.json"
        ]
        
        required_dirs = [
            self.config.base_model,
            # self.config.adapter_path,
            self.config.reward_model
        ]
        
        # 檢查訓練數據文件 - 使用統一的命名格式
        json_files = [
            f"{self.config.data_dir}/ppo_data_train.json",
            f"{self.config.data_dir}/train.json"
        ]
        
        json_found = False
        for json_file in json_files:
            if os.path.exists(json_file):
                json_found = True
                logger.info(f"Found training data: {json_file}")
                break
        
        if not json_found:
            raise FileNotFoundError(f"No training JSON file found in {self.config.data_dir}")
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
                
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Required directory not found: {dir_path}")
    
    def build_training_command(self):
        """構建訓練命令"""
        output_dir = f"{self.config.parent_dir}/ppo_training_continued_fixed"
        logging_dir = f"{self.config.parent_dir}/logs/ppo_training_continued_fixed"
        
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)
        
        env_vars = {
            "PYTHONWARNINGS": "ignore:Trainer\\.tokenizer is now deprecated:UserWarning",
            "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
            "CUDA_VISIBLE_DEVICES": self.config.gpu_devices
        }
        
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={self.config.nproc_per_node}",
            "src/train.py",
            "--stage", "ppo",
            "--model_name_or_path", self.config.base_model,
            # "--adapter_name_or_path", self.config.adapter_path,
            "--template", "qwen2_vl",
            "--finetuning_type", "lora",
            "--lora_rank", str(self.config.lora_rank),
            "--lora_alpha", str(self.config.lora_alpha),
            "--lora_dropout", str(self.config.lora_dropout),
            "--reward_model", self.config.reward_model,
            "--ref_model", self.config.ref_model,
            "--trust_remote_code", "True",
            "--do_train",
            "--dataset", self.config.dataset_name,
            "--dataset_dir", self.config.data_dir,
            "--per_device_train_batch_size", str(self.config.per_device_train_batch_size),
            "--gradient_accumulation_steps", str(self.config.gradient_accumulation_steps),
            "--learning_rate", str(self.config.learning_rate),
            "--lr_scheduler_type", "cosine",
            "--warmup_ratio", "0.1",
            "--num_train_epochs", str(self.config.num_train_epochs),
            "--logging_steps", "5",
            "--save_steps", "100",
            "--eval_steps", "100",
            "--output_dir", output_dir,
            "--logging_dir", logging_dir,
            "--run_name", self.run_name,
            "--overwrite_output_dir", "true",
            "--report_to", "tensorboard",
            "--cache_dir", "./cache",
            "--plot_loss",
            "--bf16", "true",
            "--weight_decay", "0.01",
            "--temperature", str(self.config.temperature),
            "--max_length", str(self.config.max_length),
            "--max_new_tokens", str(self.config.max_new_tokens),
            "--do_sample",
            "--seed", str(self.config.seed),
            "--save_total_limit", "3",
            "--dataloader_num_workers", "4",
            "--remove_unused_columns", "false",
            "--ddp_find_unused_parameters", "false"
        ]
        
        return cmd, env_vars
    
    def execute_training(self):
        """執行實際的訓練過程"""
        try:
            self.validate_paths()
            cmd, env_vars = self.build_training_command()
            
            # 更新環境變量
            env = os.environ.copy()
            env.update(env_vars)
            
            # 切換到工作目錄
            os.chdir(self.config.work_dir)
            
            self.start_time = datetime.now()
            self.status = "running"
            
            logger.info(f"Starting training job: {self.run_name}")
            logger.info(f"Command: {' '.join(cmd)}")
            
            # 執行命令
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # 實時讀取輸出
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "message": line.strip()
                    }
                    self.logs.append(log_entry)
                    training_status["logs"].append(log_entry)
                    logger.info(f"Training log: {line.strip()}")
            
            # 等待進程完成
            self.process.wait()
            
            self.end_time = datetime.now()
            if self.process.returncode == 0:
                self.status = "completed"
                logger.info(f"Training job completed successfully: {self.run_name}")
            else:
                self.status = "failed"
                logger.error(f"Training job failed: {self.run_name}")
                
        except Exception as e:
            self.status = "failed"
            self.end_time = datetime.now()
            error_msg = f"Training job failed with error: {str(e)}"
            logger.error(error_msg)
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "message": error_msg
            }
            self.logs.append(log_entry)
            training_status["logs"].append(log_entry)
        
        finally:
            training_status["is_running"] = False
            training_status["last_job"] = self
            training_status["current_job"] = None

def generate_dataset_info(dataset_name: str, dataset_id: str, train_data_path: str, eval_data_path: str, upload_info: Dict[str, Any]) -> Dict[str, Any]:
    """生成完整的數據集信息 - 符合 LLaMA-Factory 格式"""
    return {
        "ppo_data": {
            "file_name": "ppo_data_train.json",
            "file_path": train_data_path,
            "type": "multimodal_ppo",
            "columns": {
                "instruction": "instruction",
                "prompt": "query", 
                "response": "response",
                "labels": "labels",
                "images": "images"
            }
        },
        "ppo_data_eval": {
            "file_name": "ppo_data_eval.json", 
            "file_path": eval_data_path,
            "type": "multimodal_ppo",
            "columns": {
                "instruction": "instruction",
                "prompt": "query",
                "response": "response", 
                "labels": "labels",
                "images": "images"
            }
        },
        "_metadata": {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "upload_time": datetime.now().isoformat(),
            "file_info": upload_info
        }
    }

def parse_training_progress(logs: List[Dict[str, str]]) -> Dict[str, Any]:
    """解析訓練日志，提取進度信息"""
    progress_info = {
        "current_epoch": 0,
        "total_epochs": 0,
        "current_step": 0,
        "total_steps": 0,
        "learning_rate": 0.0,
        "loss": 0.0,
        "reward": 0.0,
        "kl_divergence": 0.0,
        "progress_percentage": 0.0,
        "estimated_time_remaining": "Unknown",
        "training_speed": "Unknown",
        "gpu_memory_usage": "Unknown",
        "latest_metrics": {},
        "loss_history": [],
        "reward_history": [],
        "status": "unknown"
    }
    
    if not logs:
        return progress_info
    
    # 解析最新的日志條目
    for log_entry in reversed(logs):
        message = log_entry.get("message", "")
        
        # 解析訓練步驟信息
        step_match = re.search(r'(\d+)/(\d+)', message)
        if step_match:
            progress_info["current_step"] = int(step_match.group(1))
            progress_info["total_steps"] = int(step_match.group(2))
            if progress_info["total_steps"] > 0:
                progress_info["progress_percentage"] = (progress_info["current_step"] / progress_info["total_steps"]) * 100
        
        # 解析 epoch 信息
        epoch_match = re.search(r'Epoch (\d+)/(\d+)', message)
        if epoch_match:
            progress_info["current_epoch"] = int(epoch_match.group(1))
            progress_info["total_epochs"] = int(epoch_match.group(2))
        
        # 解析學習率
        lr_match = re.search(r'lr[:\s=]+([0-9\.e\-]+)', message)
        if lr_match:
            progress_info["learning_rate"] = float(lr_match.group(1))
        
        # 解析損失值
        loss_match = re.search(r'loss[:\s=]+([0-9\.\-e]+)', message)
        if loss_match:
            progress_info["loss"] = float(loss_match.group(1))
        
        # 解析獎勵值 (PPO specific)
        reward_match = re.search(r'reward[:\s=]+([0-9\.\-e]+)', message)
        if reward_match:
            progress_info["reward"] = float(reward_match.group(1))
        
        # 解析 KL 散度 (PPO specific)
        kl_match = re.search(r'kl[:\s=]+([0-9\.\-e]+)', message)
        if kl_match:
            progress_info["kl_divergence"] = float(kl_match.group(1))
        
        # 解析訓練速度
        speed_match = re.search(r'(\d+\.\d+)\s*it/s', message)
        if speed_match:
            progress_info["training_speed"] = f"{speed_match.group(1)} it/s"
        
        # 解析 GPU 內存使用
        gpu_match = re.search(r'GPU.*?(\d+)MiB', message)
        if gpu_match:
            progress_info["gpu_memory_usage"] = f"{gpu_match.group(1)}MiB"
    
    # 構建損失和獎勵歷史
    for log_entry in logs[-50:]:  # 只保留最近50條記錄
        message = log_entry.get("message", "")
        timestamp = log_entry.get("timestamp", "")
        
        loss_match = re.search(r'loss[:\s=]+([0-9\.\-e]+)', message)
        if loss_match:
            progress_info["loss_history"].append({
                "timestamp": timestamp,
                "step": progress_info["current_step"],
                "loss": float(loss_match.group(1))
            })
        
        reward_match = re.search(r'reward[:\s=]+([0-9\.\-e]+)', message)
        if reward_match:
            progress_info["reward_history"].append({
                "timestamp": timestamp,
                "step": progress_info["current_step"],
                "reward": float(reward_match.group(1))
            })
    
    # 確定訓練狀態
    if training_status["is_running"]:
        progress_info["status"] = "running"
    elif training_status["last_job"]:
        progress_info["status"] = training_status["last_job"].status
    else:
        progress_info["status"] = "idle"
    
    return progress_info

# 添加訓練進度解析函數
def parse_training_progress(logs: List[Dict[str, str]]) -> Dict[str, Any]:
    """解析訓練日志，提取進度信息"""
    progress_info = {
        "current_epoch": 0,
        "total_epochs": 0,
        "current_step": 0,
        "total_steps": 0,
        "learning_rate": 0.0,
        "loss": 0.0,
        "reward": 0.0,
        "kl_divergence": 0.0,
        "progress_percentage": 0.0,
        "estimated_time_remaining": "Unknown",
        "training_speed": "Unknown",
        "gpu_memory_usage": "Unknown",
        "latest_metrics": {},
        "loss_history": [],
        "reward_history": [],
        "status": "unknown"
    }
    
    if not logs:
        return progress_info
    
    # 解析最新的日志條目
    for log_entry in reversed(logs):
        message = log_entry.get("message", "")
        
        # 解析訓練步驟信息
        step_match = re.search(r'(\d+)/(\d+)', message)
        if step_match:
            progress_info["current_step"] = int(step_match.group(1))
            progress_info["total_steps"] = int(step_match.group(2))
            if progress_info["total_steps"] > 0:
                progress_info["progress_percentage"] = (progress_info["current_step"] / progress_info["total_steps"]) * 100
        
        # 解析 epoch 信息
        epoch_match = re.search(r'Epoch (\d+)/(\d+)', message)
        if epoch_match:
            progress_info["current_epoch"] = int(epoch_match.group(1))
            progress_info["total_epochs"] = int(epoch_match.group(2))
        
        # 解析學習率
        lr_match = re.search(r'lr[:\s=]+([0-9\.e\-]+)', message)
        if lr_match:
            progress_info["learning_rate"] = float(lr_match.group(1))
        
        # 解析損失值
        loss_match = re.search(r'loss[:\s=]+([0-9\.\-e]+)', message)
        if loss_match:
            progress_info["loss"] = float(loss_match.group(1))
        
        # 解析獎勵值 (PPO specific)
        reward_match = re.search(r'reward[:\s=]+([0-9\.\-e]+)', message)
        if reward_match:
            progress_info["reward"] = float(reward_match.group(1))
        
        # 解析 KL 散度 (PPO specific)
        kl_match = re.search(r'kl[:\s=]+([0-9\.\-e]+)', message)
        if kl_match:
            progress_info["kl_divergence"] = float(kl_match.group(1))
        
        # 解析訓練速度
        speed_match = re.search(r'(\d+\.\d+)\s*it/s', message)
        if speed_match:
            progress_info["training_speed"] = f"{speed_match.group(1)} it/s"
        
        # 解析 GPU 內存使用
        gpu_match = re.search(r'GPU.*?(\d+)MiB', message)
        if gpu_match:
            progress_info["gpu_memory_usage"] = f"{gpu_match.group(1)}MiB"
    
    # 構建損失和獎勵歷史
    for log_entry in logs[-50:]:  # 只保留最近50條記錄
        message = log_entry.get("message", "")
        timestamp = log_entry.get("timestamp", "")
        
        loss_match = re.search(r'loss[:\s=]+([0-9\.\-e]+)', message)
        if loss_match:
            progress_info["loss_history"].append({
                "timestamp": timestamp,
                "step": progress_info["current_step"],
                "loss": float(loss_match.group(1))
            })
        
        reward_match = re.search(r'reward[:\s=]+([0-9\.\-e]+)', message)
        if reward_match:
            progress_info["reward_history"].append({
                "timestamp": timestamp,
                "step": progress_info["current_step"],
                "reward": float(reward_match.group(1))
            })
    
    # 確定訓練狀態
    if training_status["is_running"]:
        progress_info["status"] = "running"
    elif training_status["last_job"]:
        progress_info["status"] = training_status["last_job"].status
    else:
        progress_info["status"] = "idle"
    
    return progress_info

# 添加新的 API 端點

@app.get("/training/status")
async def get_training_status():
    """獲取當前訓練狀態概覽"""
    current_job = training_status.get("current_job")
    last_job = training_status.get("last_job")
    
    status_info = {
        "is_running": training_status["is_running"],
        "current_job": None,
        "last_job": None,
        "total_logs": len(training_status["logs"])
    }
    
    if current_job:
        status_info["current_job"] = {
            "run_name": current_job.run_name,
            "status": current_job.status,
            "start_time": current_job.start_time.isoformat() if current_job.start_time else None,
            "dataset_name": current_job.config.dataset_name,
            "data_dir": current_job.config.data_dir
        }
    
    if last_job:
        status_info["last_job"] = {
            "run_name": last_job.run_name,
            "status": last_job.status,
            "start_time": last_job.start_time.isoformat() if last_job.start_time else None,
            "end_time": last_job.end_time.isoformat() if last_job.end_time else None,
            "dataset_name": last_job.config.dataset_name
        }
    
    return status_info

@app.get("/training/progress")
async def get_training_progress():
    """獲取詳細的訓練進度信息"""
    if not training_status["is_running"] and not training_status["last_job"]:
        raise HTTPException(status_code=404, detail="No training job found")
    
    current_job = training_status.get("current_job") or training_status.get("last_job")
    logs = training_status.get("logs", [])
    
    # 解析訓練進度
    progress = parse_training_progress(logs)
    
    # 添加作業信息
    job_info = {
        "run_name": current_job.run_name if current_job else "Unknown",
        "status": current_job.status if current_job else "unknown",
        "start_time": current_job.start_time.isoformat() if current_job and current_job.start_time else None,
        "end_time": current_job.end_time.isoformat() if current_job and current_job.end_time else None,
        "dataset_name": current_job.config.dataset_name if current_job else "Unknown",
        "config": current_job.config.dict() if current_job else {}
    }
    
    # 計算運行時間
    if current_job and current_job.start_time:
        if current_job.end_time:
            duration = (current_job.end_time - current_job.start_time).total_seconds()
        else:
            duration = (datetime.now() - current_job.start_time).total_seconds()
        
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        job_info["runtime"] = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        
        # 估算剩余時間
        if progress["progress_percentage"] > 0 and training_status["is_running"]:
            estimated_total_time = duration / (progress["progress_percentage"] / 100)
            remaining_time = estimated_total_time - duration
            if remaining_time > 0:
                rem_hours, rem_remainder = divmod(remaining_time, 3600)
                rem_minutes, rem_seconds = divmod(rem_remainder, 60)
                progress["estimated_time_remaining"] = f"{int(rem_hours):02d}:{int(rem_minutes):02d}:{int(rem_seconds):02d}"
    
    return {
        "job_info": job_info,
        "progress": progress,
        "recent_logs": logs[-10:] if logs else [],  # 最近10條日志
        "log_count": len(logs)
    }

@app.get("/training/metrics")
async def get_training_metrics():
    """獲取訓練指標和圖表數據"""
    logs = training_status.get("logs", [])
    
    if not logs:
        raise HTTPException(status_code=404, detail="No training logs available")
    
    # 提取指標數據用於圖表展示
    metrics = {
        "loss_data": [],
        "reward_data": [],  # PPO specific
        "kl_divergence_data": [],  # PPO specific
        "learning_rate_data": [],
        "step_time_data": [],
        "memory_usage_data": []
    }
    
    step_counter = 0
    for log_entry in logs:
        message = log_entry.get("message", "")
        timestamp = log_entry.get("timestamp", "")
        
        # 如果檢測到步驟信息，增加計數器
        if re.search(r'(\d+)/(\d+)', message):
            step_counter += 1
        
        # 提取各種指標
        loss_match = re.search(r'loss[:\s=]+([0-9\.\-e]+)', message)
        if loss_match:
            metrics["loss_data"].append({
                "step": step_counter,
                "timestamp": timestamp,
                "value": float(loss_match.group(1))
            })
        
        reward_match = re.search(r'reward[:\s=]+([0-9\.\-e]+)', message)
        if reward_match:
            metrics["reward_data"].append({
                "step": step_counter,
                "timestamp": timestamp,
                "value": float(reward_match.group(1))
            })
        
        kl_match = re.search(r'kl[:\s=]+([0-9\.\-e]+)', message)
        if kl_match:
            metrics["kl_divergence_data"].append({
                "step": step_counter,
                "timestamp": timestamp,
                "value": float(kl_match.group(1))
            })
        
        lr_match = re.search(r'lr[:\s=]+([0-9\.e\-]+)', message)
        if lr_match:
            metrics["learning_rate_data"].append({
                "step": step_counter,
                "timestamp": timestamp,
                "value": float(lr_match.group(1))
            })
    
    # 計算統計信息
    stats = {}
    for metric_name, data in metrics.items():
        if data:
            values = [item["value"] for item in data]
            stats[metric_name] = {
                "count": len(values),
                "latest": values[-1],
                "min": min(values),
                "max": max(values),
                "average": sum(values) / len(values)
            }
    
    return {
        "metrics": metrics,
        "statistics": stats,
        "data_points": sum(len(data) for data in metrics.values())
    }

@app.get("/training/logs/stream")
async def stream_training_logs(last_log_index: Optional[int] = None):
    """獲取增量訓練日志 (用於實時更新)"""
    logs = training_status.get("logs", [])
    
    if last_log_index is None:
        # 返回最近50條日志
        return {
            "logs": logs[-50:],
            "total_logs": len(logs),
            "last_index": len(logs) - 1 if logs else -1
        }
    else:
        # 返回指定索引之後的新日志
        new_logs = logs[last_log_index + 1:] if last_log_index >= 0 else logs
        return {
            "logs": new_logs,
            "total_logs": len(logs),
            "last_index": len(logs) - 1 if logs else -1,
            "new_logs_count": len(new_logs)
        }

# 修改現有的日志端點，添加更多過濾選項
@app.get("/logs/filtered")
async def get_filtered_logs(
    limit: int = 100,
    level: Optional[str] = None,  # error, warning, info
    contains: Optional[str] = None,  # 包含特定文本
    from_step: Optional[int] = None,  # 從特定步驟開始
    to_step: Optional[int] = None     # 到特定步驟結束
):
    """獲取過濾後的訓練日志"""
    logs = training_status["logs"]
    filtered_logs = logs.copy()
    
    # 按級別過濾
    if level:
        level_keywords = {
            "error": ["error", "failed", "exception"],
            "warning": ["warning", "warn"],
            "info": ["info", "step", "epoch", "loss"]
        }
        keywords = level_keywords.get(level.lower(), [])
        if keywords:
            filtered_logs = [
                log for log in filtered_logs 
                if any(keyword in log.get("message", "").lower() for keyword in keywords)
            ]
    
    # 按內容過濾
    if contains:
        filtered_logs = [
            log for log in filtered_logs 
            if contains.lower() in log.get("message", "").lower()
        ]
    
    # 按步驟範圍過濾
    if from_step is not None or to_step is not None:
        step_filtered_logs = []
        for log in filtered_logs:
            message = log.get("message", "")
            step_match = re.search(r'(\d+)/(\d+)', message)
            if step_match:
                current_step = int(step_match.group(1))
                if from_step is not None and current_step < from_step:
                    continue
                if to_step is not None and current_step > to_step:
                    continue
            step_filtered_logs.append(log)
        filtered_logs = step_filtered_logs
    
    # 應用限制
    if limit > 0:
        filtered_logs = filtered_logs[-limit:]
    
    return {
        "logs": filtered_logs,
        "total_filtered": len(filtered_logs),
        "total_logs": len(logs),
        "filters_applied": {
            "level": level,
            "contains": contains,
            "from_step": from_step,
            "to_step": to_step,
            "limit": limit
        }
    }
    

def run_training_in_background(job: TrainingJob):
    """在背景執行訓練工作"""
    training_status["is_running"] = True
    training_status["current_job"] = job
    training_status["logs"] = []
    
    job.execute_training()

def process_csv_data(csv_file_path: str, dataset_name: str, dataset_id: str, data_dir: str, original_filename: str) -> Dict[str, Any]:
    """處理 CSV 檔案並轉換為訓練格式"""
    
    # 讀取 CSV
    df = pd.read_csv(csv_file_path)
    
    # 檢查必要的欄位
    required_columns = ['images_path', 'question', 'answer']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV must contain column: {col}")
    
    # 確保目標目錄存在
    os.makedirs(data_dir, exist_ok=True)
    target_image_dir = os.path.join(data_dir, 'images')
    os.makedirs(target_image_dir, exist_ok=True)
    
    # 轉換資料格式
    processed_data = []
    image_files_copied = 0
    
    for _, row in df.iterrows():
        # 複製圖片檔案
        original_image_path = row['images_path']
        if os.path.exists(original_image_path):
            image_filename = os.path.basename(original_image_path)
            target_image_path = os.path.join(target_image_dir, image_filename)
            
            # 複製圖片
            shutil.copy2(original_image_path, target_image_path)
            image_files_copied += 1
            
            # 建立訓練資料格式
            training_item = {
                "instruction": "You are a knowledgeable assistant with expertise in quantitative analysis and visual understanding. Please carefully analyze images and respond to all types of questions, including counting objects, describing scenes, explaining content, or answering general knowledge queries. Answer in Traditional Chinese.",
                "query": f"<image>{row['question']}",
                "images": [target_image_path],
                "response": str(row['answer']),
                "labels": str(row['answer'])
            }
            processed_data.append(training_item)
        else:
            logger.warning(f"Image file not found: {original_image_path}")
    
    # 分割資料為訓練集和評估集 (80/20 分割)
    train_data, eval_data = train_test_split(processed_data, test_size=0.2, random_state=42)
    
    # 使用固定的檔案名稱
    train_data_path = os.path.join(data_dir, 'ppo_data_train.json')
    with open(train_data_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    # 儲存評估資料
    eval_data_path = os.path.join(data_dir, 'ppo_data_eval.json')
    with open(eval_data_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    
    # 建立 dataset_info.json
    upload_info = {
        "original_filename": original_filename,
        "file_type": "CSV",
        "csv_rows": len(df),
        "training_samples": len(train_data),
        "evaluation_samples": len(eval_data),
        "images_copied": image_files_copied
    }
    
    dataset_info = generate_dataset_info(dataset_name, dataset_id, train_data_path, eval_data_path, upload_info)
    
    dataset_info_path = os.path.join(data_dir, 'dataset_info.json')
    with open(dataset_info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    # 生成訓練配置檔案
    training_config = {
        "work_dir": "/home/itrib30156/llm_vision/LLaMA-Factory/LLaMA-Factory",
        "parent_dir": "/home/itrib30156/llm_vision/LLaMA-Factory",
        "data_dir": data_dir,
        "base_model": "/home/itrib30156/llm_vision/phi4",
        # "adapter_path": "/home/itrib30156/llm_vision/LLaMA-Factory/ppo_model/checkpoint-650",
        "reward_model": "/home/itrib30156/llm_vision/LLaMA-Factory/rm_model/checkpoint-600",
        "ref_model": "/home/itrib30156/llm_vision/phi4",
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "learning_rate": 0.000005,
        "num_train_epochs": 1,
        "lora_rank": 64,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "max_length": 512,
        "max_new_tokens": 128,
        "temperature": 0.7,
        "seed": 42,
        "gpu_devices": "0,1",
        "nproc_per_node": 2,
        "dataset_name": "ppo_data"
    }
    
    config_path = os.path.join(data_dir, 'training_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(training_config, f, ensure_ascii=False, indent=2)
    
    return {
        "dataset_info": dataset_info,
        "files_processed": upload_info
    }

def process_zip_data(zip_file: UploadFile, dataset_name: str, dataset_id: str, data_dir: str, temp_dir: str, original_filename: str) -> Dict[str, Any]:
    """處理上傳的 ZIP 檔案 (包含 CSV 和 images 資料夾)"""
    
    # 儲存上傳的檔案
    zip_path = f"{temp_dir}/uploaded_data.zip"
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(zip_file.file, buffer)
    
    # 解壓縮
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # 查找 CSV 檔案和圖片目錄
    csv_files = []
    image_dirs = []
    
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
        for dir_name in dirs:
            if 'image' in dir_name.lower() or 'img' in dir_name.lower():
                image_dirs.append(os.path.join(root, dir_name))
    
    if not csv_files:
        raise ValueError("No CSV files found in uploaded ZIP")
    
    if not image_dirs:
        raise ValueError("No images directory found in uploaded ZIP")
    
    # 使用第一個找到的 CSV 檔案
    csv_file_path = csv_files[0]
    source_image_dir = image_dirs[0]
    
    # 讀取 CSV
    df = pd.read_csv(csv_file_path)
    
    # 檢查必要的欄位
    required_columns = ['images_path', 'question', 'answer']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV must contain column: {col}")
    
    # 確保目標目錄存在
    os.makedirs(data_dir, exist_ok=True)
    target_image_dir = os.path.join(data_dir, 'images')
    
    # 複製整個圖片目錄
    if os.path.exists(target_image_dir):
        shutil.rmtree(target_image_dir)
    shutil.copytree(source_image_dir, target_image_dir)
    
    # 轉換資料格式
    processed_data = []
    image_files_found = 0
    image_files_missing = 0
    
    for _, row in df.iterrows():
        # 從原始路徑提取檔案名稱
        original_image_path = row['images_path']
        image_filename = os.path.basename(original_image_path)
        target_image_path = os.path.join(target_image_dir, image_filename)
        
        # 檢查圖片是否存在
        if os.path.exists(target_image_path):
            image_files_found += 1
            
            # 建立訓練資料格式
            training_item = {
                "instruction": "You are a knowledgeable assistant with expertise in quantitative analysis and visual understanding. Please carefully analyze images and respond to all types of questions, including counting objects, describing scenes, explaining content, or answering general knowledge queries. Answer in Traditional Chinese.",
                "query": f"<image>{row['question']}",
                "images": [target_image_path],
                "response": str(row['answer']),
                "labels": str(row['answer'])
            }
            processed_data.append(training_item)
        else:
            image_files_missing += 1
            logger.warning(f"Image file not found in ZIP: {image_filename}")
    
    # 分割資料為訓練集和評估集 (80/20 分割)  
    train_data, eval_data = train_test_split(processed_data, test_size=0.2, random_state=42)
    
    # 使用固定的檔案名稱
    train_data_path = os.path.join(data_dir, 'ppo_data_train.json')
    with open(train_data_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    # 儲存評估資料
    eval_data_path = os.path.join(data_dir, 'ppo_data_eval.json')
    with open(eval_data_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    
    # 建立 dataset_info.json (LLaMA-Factory 標準格式)
    upload_info = {
        "original_filename": original_filename,
        "file_type": "ZIP",
        "csv_rows": len(df),
        "training_samples": len(train_data),
        "evaluation_samples": len(eval_data),
        "images_found": image_files_found,
        "images_missing": image_files_missing,
        "total_images": len([f for f in os.listdir(target_image_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))])
    }
    
    dataset_info = generate_dataset_info(dataset_name, dataset_id, train_data_path, eval_data_path, upload_info)
    
    dataset_info_path = os.path.join(data_dir, 'dataset_info.json')
    with open(dataset_info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    # 生成訓練配置檔案
    training_config = {
        "work_dir": "/home/itrib30156/llm_vision/LLaMA-Factory/LLaMA-Factory",
        "parent_dir": "/home/itrib30156/llm_vision/LLaMA-Factory",
        "data_dir": data_dir,
        "base_model": "/home/itrib30156/llm_vision/phi4",
        # "adapter_path": "/home/itrib30156/llm_vision/LLaMA-Factory/ppo_model/checkpoint-650",
        "reward_model": "/home/itrib30156/llm_vision/LLaMA-Factory/rm_model/checkpoint-600",
        "ref_model": "/home/itrib30156/llm_vision/phi4",
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "learning_rate": 0.000005,
        "num_train_epochs": 1,
        "lora_rank": 64,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "max_length": 512,
        "max_new_tokens": 128,
        "temperature": 0.7,
        "seed": 42,
        "gpu_devices": "0,1",
        "nproc_per_node": 2,
        "dataset_name": "ppo_data"
    }
    
    config_path = os.path.join(data_dir, 'training_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(training_config, f, ensure_ascii=False, indent=2)
    
    # 刪除原始 ZIP 檔案 - 圖片已經複製到最終位置了
    if os.path.exists(zip_path):
        os.remove(zip_path)
        logger.info(f"Deleted original ZIP file: {zip_path}")
    
    return {
        "dataset_info": dataset_info,
        "files_processed": upload_info
    }

def process_uploaded_data(file: UploadFile, dataset_name: str, dataset_id: str, data_dir: str) -> Dict[str, Any]:
    """處理上傳的檔案 (支援 ZIP 和 CSV)"""
    
    # 使用臨時目錄
    temp_dir = f"/home/itrib30156/llm_vision/LLaMA-Factory/temp/upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        filename = file.filename.lower()
        original_filename = file.filename
        
        # 如果是 CSV 檔案
        if filename.endswith('.csv'):
            csv_path = f"{temp_dir}/data.csv"
            with open(csv_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            return process_csv_data(csv_path, dataset_name, dataset_id, data_dir, original_filename)
        
        # 如果是 ZIP 檔案
        elif filename.endswith('.zip'):
            return process_zip_data(file, dataset_name, dataset_id, data_dir, temp_dir, original_filename)
        
        else:
            raise ValueError("Only CSV and ZIP files are supported")
    
    finally:
        # 清理臨時目錄
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")

# API 端點
@app.get("/")
async def root():
    return {"message": "PPO Training API with Data Upload is running"}

@app.get("/data/{dataset_id}/config")
async def get_dataset_config(dataset_id: str):
    """獲取數據集的當前訓練配置"""
    data_dir = f"/home/itrib30156/llm_vision/LLaMA-Factory/data/{dataset_id}"
    
    if not os.path.exists(data_dir):
        raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found")
    
    config_path = os.path.join(data_dir, 'training_config.json')
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            # 使用默認配置
            config = {
                "work_dir": "/home/itrib30156/llm_vision/LLaMA-Factory/LLaMA-Factory",
                "parent_dir": "/home/itrib30156/llm_vision/LLaMA-Factory",
                "data_dir": data_dir,
                "base_model": "/home/itrib30156/llm_vision/phi4",
                # "adapter_path": "/home/itrib30156/llm_vision/LLaMA-Factory/ppo_model/checkpoint-650",
                "reward_model": "/home/itrib30156/llm_vision/LLaMA-Factory/rm_model/checkpoint-600",
                "ref_model": "/home/itrib30156/llm_vision/phi4",
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 2,
                "learning_rate": 0.000005,
                "num_train_epochs": 1,
                "lora_rank": 64,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "max_length": 512,
                "max_new_tokens": 128,
                "temperature": 0.7,
                "seed": 42,
                "gpu_devices": "0,1",
                "nproc_per_node": 2,
                "dataset_name": "ppo_data"
            }
        
        return {
            "dataset_id": dataset_id,
            "config": config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read configuration: {str(e)}")

@app.get("/logs")
async def get_training_logs(limit: int = 100):
    """獲取訓練日誌"""
    logs = training_status["logs"][-limit:] if limit > 0 else training_status["logs"]
    return {"logs": logs, "total_count": len(training_status["logs"])}

@app.post("/training/start_with_config")
async def start_training_with_full_config(
    dataset_id: str = Form(...),
    background_tasks: BackgroundTasks = None,
    work_dir: str = Form("/home/itrib30156/llm_vision/LLaMA-Factory/LLaMA-Factory"),
    parent_dir: str = Form("/home/itrib30156/llm_vision/LLaMA-Factory"),
    base_model: str = Form("/home/itrib30156/llm_vision/phi4"),
    # adapter_path: str = Form("/home/itrib30156/llm_vision/LLaMA-Factory/ppo_model/checkpoint-650"),
    reward_model: str = Form("/home/itrib30156/llm_vision/LLaMA-Factory/rm_model/checkpoint-600"),
    ref_model: str = Form("/home/itrib30156/llm_vision/phi4"),
    per_device_train_batch_size: int = Form(2),
    gradient_accumulation_steps: int = Form(2),
    learning_rate: float = Form(0.000005),
    num_train_epochs: int = Form(1),
    lora_rank: int = Form(64),
    lora_alpha: int = Form(32),
    lora_dropout: float = Form(0.1),
    max_length: int = Form(512),
    max_new_tokens: int = Form(128),
    temperature: float = Form(0.7),
    seed: int = Form(42),
    gpu_devices: str = Form("0,1"),
    nproc_per_node: int = Form(2),
    dataset_name: str = Form("ppo_data")
):
    """啟動 PPO 訓練 - 使用完整配置（帶有默認值）"""
    if training_status["is_running"]:
        raise HTTPException(status_code=400, detail="Training is already running")
    
    # 檢查數據集是否存在
    data_dir = f"/home/itrib30156/llm_vision/LLaMA-Factory/data/{dataset_id}"
    if not os.path.exists(data_dir):
        raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found")
    
    try:
        # 構建配置
        config_dict = {
            "work_dir": work_dir,
            "parent_dir": parent_dir,
            "data_dir": data_dir,
            "base_model": base_model,
            # "adapter_path": adapter_path,
            "reward_model": reward_model,
            "ref_model": ref_model,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "max_length": max_length,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "seed": seed,
            "gpu_devices": gpu_devices,
            "nproc_per_node": nproc_per_node,
            "dataset_name": dataset_name
        }
        
        # 創建配置對象
        config = TrainingConfig(**config_dict)
        
        # 保存當前配置到數據集
        config_path = os.path.join(data_dir, 'training_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        
        job = TrainingJob(config)
        background_tasks.add_task(run_training_in_background, job)
        
        return {
            "message": "PPO training started successfully",
            "run_name": job.run_name,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "config": config.dict(),
            "data_files": {
                "train": f"{data_dir}/ppo_data_train.json",
                "eval": f"{data_dir}/ppo_data_eval.json"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to start training: {str(e)}")
    

@app.post("/data/upload")
async def upload_training_data(
    file: UploadFile = File(...),
    dataset_name: str = Form(None),
    description: str = Form("")
) -> DataUploadResponse:
    """上傳訓練資料檔案 (支援 CSV 直接上傳或 ZIP 包含 CSV+images)"""
    
    filename = file.filename.lower()
    if not (filename.endswith('.csv') or filename.endswith('.zip')):
        raise HTTPException(status_code=400, detail="Only CSV and ZIP files are supported")
    
    # 生成唯一 ID
    dataset_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 自動生成資料夾名稱
    if dataset_name is None:
        # 使用檔案名稱（去除副檔名）作為基礎名稱
        base_name = os.path.splitext(file.filename)[0]
        dataset_name = f"{base_name}_{timestamp}"
    else:
        # 如果指定了 dataset_name，加上時間戳避免重複
        dataset_name = f"{dataset_name}_{timestamp}"
    
    # 使用 UUID 作為目錄名稱，確保唯一性
    data_dir = f"/home/itrib30156/llm_vision/LLaMA-Factory/data/{dataset_id}"
    
    try:
        result = process_uploaded_data(file, dataset_name, dataset_id, data_dir)
        
        file_type = "CSV" if filename.endswith('.csv') else "ZIP (CSV + images)"
        upload_time = datetime.now().isoformat()
        
        return DataUploadResponse(
            message=f"Successfully uploaded and processed {file_type} training data in LLaMA-Factory format",
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            dataset_info=result["dataset_info"],
            files_processed=result["files_processed"],
            upload_time=upload_time
        )
        
    except Exception as e:
        logger.error(f"Failed to process uploaded data: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to process uploaded data: {str(e)}")

@app.post("/training/stop")
async def stop_ppo_training():
    """停止 PPO 訓練"""
    if not training_status["is_running"]:
        raise HTTPException(status_code=400, detail="No training is currently running")
    
    current_job = training_status["current_job"]
    if current_job and current_job.process:
        current_job.process.terminate()
        current_job.status = "stopped"
        current_job.end_time = datetime.now()
        
        training_status["is_running"] = False
        training_status["last_job"] = current_job
        training_status["current_job"] = None
        
        return {"message": "PPO training stopped successfully"}
    else:
        raise HTTPException(status_code=400, detail="Unable to stop training process")


@app.get("/data/list")
async def list_datasets():
    """列出可用的數據集"""
    data_base_dir = "/home/itrib30156/llm_vision/LLaMA-Factory/data"
    datasets = []
    
    if os.path.exists(data_base_dir):
        for item in os.listdir(data_base_dir):
            item_path = os.path.join(data_base_dir, item)
            if os.path.isdir(item_path):
                dataset_info_path = os.path.join(item_path, 'dataset_info.json')
                if os.path.exists(dataset_info_path):
                    try:
                        with open(dataset_info_path, 'r', encoding='utf-8') as f:
                            info = json.load(f)
                        
                        # 獲取元數據
                        metadata = info.get("metadata", {})
                        dataset_id = metadata.get("dataset_id", item)
                        dataset_name = metadata.get("dataset_name", "Unknown")
                        upload_time = metadata.get("upload_time", "Unknown")
                        file_info = metadata.get("file_info", {})
                        
                        # 獲取資料夾統計信息
                        train_samples = file_info.get("training_samples", 0)
                        eval_samples = file_info.get("evaluation_samples", 0)
                        image_count = file_info.get("total_images", file_info.get("images_copied", 0))
                        
                        # 計算圖片數量（如果統計信息不存在）
                        if image_count == 0:
                            images_dir = os.path.join(item_path, 'images')
                            if os.path.exists(images_dir):
                                image_count = len([f for f in os.listdir(images_dir) 
                                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))])
                        
                        datasets.append({
                            "dataset_id": dataset_id,
                            "dataset_name": dataset_name,
                            "path": item_path,
                            "upload_time": upload_time,
                            "file_info": file_info,
                            "statistics": {
                                "training_samples": train_samples,
                                "evaluation_samples": eval_samples,
                                "total_samples": train_samples + eval_samples,
                                "images": image_count,
                                "created_time": datetime.fromtimestamp(os.path.getctime(item_path)).isoformat()
                            }
                        })
                    except Exception as e:
                        logger.warning(f"Failed to read dataset info for {item}: {e}")
    
    return {"datasets": datasets, "total_datasets": len(datasets)}

@app.get("/data/{dataset_id}")
async def get_dataset_info(dataset_id: str):
    """獲取特定數據集的詳細信息"""
    data_dir = f"/home/itrib30156/llm_vision/LLaMA-Factory/data/{dataset_id}"
    
    if not os.path.exists(data_dir):
        raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found")
    
    dataset_info_path = os.path.join(data_dir, 'dataset_info.json')
    if not os.path.exists(dataset_info_path):
        raise HTTPException(status_code=404, detail=f"Dataset info file not found for ID {dataset_id}")
    
    try:
        with open(dataset_info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        
        # 獲取實際檔案統計
        images_dir = os.path.join(data_dir, 'images')
        actual_image_count = 0
        if os.path.exists(images_dir):
            actual_image_count = len([f for f in os.listdir(images_dir) 
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))])
        
        return {
            "dataset_info": info,
            "actual_statistics": {
                "images_on_disk": actual_image_count,
                "directory_size": sum(os.path.getsize(os.path.join(data_dir, f)) 
                                    for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read dataset info: {str(e)}")

@app.delete("/data/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """刪除指定的數據集"""
    data_dir = f"/home/itrib30156/llm_vision/LLaMA-Factory/data/{dataset_id}"
    
    if not os.path.exists(data_dir):
        raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found")
    
    try:
        # 獲取數據集信息用於回應
        dataset_info_path = os.path.join(data_dir, 'dataset_info.json')
        dataset_name = "Unknown"
        if os.path.exists(dataset_info_path):
            with open(dataset_info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
                dataset_name = info.get("_metadata", {}).get("dataset_name", "Unknown")
        
        # 刪除整個資料夾
        shutil.rmtree(data_dir)
        logger.info(f"Deleted dataset: {dataset_id} ({dataset_name})")
        
        return {
            "message": f"Successfully deleted dataset {dataset_name}",
            "dataset_id": dataset_id,
            "dataset_name": dataset_name
        }
    except Exception as e:
        logger.error(f"Failed to delete dataset {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)