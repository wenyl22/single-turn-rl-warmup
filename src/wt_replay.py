import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from datetime import datetime
import textwrap

class DetailedGameLogReplayer:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.config = self.load_config()
        self.game_log = self.load_game_log()
        self.streaming_log = self.load_streaming_log()
        
        # 将所有事件按时间排序
        self.all_events = self.merge_and_sort_events()
        
    def load_config(self):
        with open(f"{self.log_dir}/config.json", "r") as f:
            return json.load(f)
    
    def load_game_log(self):
        with open(f"{self.log_dir}/game_log.json", "r") as f:
            return json.load(f)
    
    def load_streaming_log(self):
        with open(f"{self.log_dir}/streaming_log.json", "r") as f:
            return json.load(f)
    
    def merge_and_sort_events(self):
        """合并并按时间排序所有事件"""
        all_events = []
        
        # 添加游戏日志事件
        for event in self.game_log:
            event['source'] = 'game'
            all_events.append(event)
        
        # 添加流式日志事件
        for event in self.streaming_log:
            event['source'] = 'streaming'
            all_events.append(event)
        
        # 按时间戳排序
        all_events.sort(key=lambda x: x['timestamp'])
        return all_events
    
    def generate_streaming_video(self, output_file="detailed_game_replay.mp4", fps=10, time_acceleration=10):
        """生成详细的流式回放视频"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        
        # 准备数据结构来跟踪状态
        current_state = {
            'turn': 0,
            'fast_content': '',
            'slow_content': '',
            'game_state': '',
            'current_action': '',
            'reward': 0,
            'reasoning_mode': False
        }
        
        def animate(frame_idx):
            if frame_idx >= len(self.all_events):
                return
            
            event = self.all_events[frame_idx]
            
            # 更新当前状态
            self.update_current_state(current_state, event)
            
            # 清除所有子图
            for ax in [ax1, ax2, ax3, ax4]:
                ax.clear()
            
            # 左上：游戏状态和回合信息
            self.plot_game_status(ax1, event, current_state)
            
            # 右上：Fast Agent输出
            self.plot_agent_output(ax2, current_state['fast_content'], "Fast Agent", 'blue')
            
            # 左下：Slow Agent输出
            self.plot_agent_output(ax3, current_state['slow_content'], "Slow Agent", 'red', current_state['reasoning_mode'])
            
            # 右下：时间线和统计
            self.plot_timeline(ax4, frame_idx, event)
            
            plt.tight_layout()
        
        # 创建动画
        anim = animation.FuncAnimation(
            fig, animate, frames=len(self.all_events),
            interval=1000/fps, repeat=False
        )
        
        # 保存视频
        anim.save(output_file, writer='ffmpeg', fps=fps, bitrate=1800)
        print(f"Detailed streaming video saved to: {output_file}")
    
    def update_current_state(self, state, event):
        """根据事件更新当前状态"""
        if event['source'] == 'game':
            if event['type'] == 'turn_start':
                state['turn'] = event['turn']
                state['game_state'] = str(event.get('state_for_llm', ''))
            elif event['type'] == 'action_execution':
                state['current_action'] = event['action']
                state['reward'] = event['reward']
        
        elif event['source'] == 'streaming':
            if event['agent'] == 'fast':
                state['fast_content'] = event['total_content']
            elif event['agent'] == 'slow':
                state['slow_content'] = event['total_content']
                state['reasoning_mode'] = event.get('reasoning', False)
    
    def plot_game_status(self, ax, event, state):
        """绘制游戏状态"""
        status_text = f"""
Turn: {state['turn']}
Current Action: {state['current_action']}
Reward: {state['reward']}
Time: {datetime.fromtimestamp(event['timestamp']).strftime('%H:%M:%S.%f')[:-3]}
Relative Time: {event['relative_time']:.2f}s

Event Type: {event['type']}
Source: {event['source']}
        """
        
        if event['source'] == 'streaming':
            status_text += f"\nAgent: {event['agent']}"
            status_text += f"\nContent Length: {event['content_length']}"
            if 'reasoning' in event:
                status_text += f"\nReasoning Mode: {event['reasoning']}"
        
        ax.text(0.05, 0.95, status_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax.set_title("Game Status", fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def plot_agent_output(self, ax, content, agent_name, color, reasoning_mode=False):
        """绘制智能体输出"""
        # 处理太长的文本
        display_content = content[-1000:] if len(content) > 1000 else content
        
        # 添加换行处理
        wrapped_content = textwrap.fill(display_content, width=60)
        
        # 设置背景色以区分推理模式
        bg_color = 'lightyellow' if reasoning_mode else 'white'
        
        ax.text(0.05, 0.95, wrapped_content, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=bg_color, alpha=0.7))
        
        title = f"{agent_name} {'(Reasoning)' if reasoning_mode else ''}"
        ax.set_title(title, fontsize=12, fontweight='bold', color=color)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def plot_timeline(self, ax, frame_idx, current_event):
        """绘制时间线和统计信息"""
        # 统计不同类型事件的数量
        event_counts = {}
        for i in range(min(frame_idx + 1, len(self.all_events))):
            event_type = self.all_events[i]['type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # 绘制统计条形图
        types = list(event_counts.keys())
        counts = list(event_counts.values())
        
        if types:
            bars = ax.bar(range(len(types)), counts, alpha=0.7)
            ax.set_xticks(range(len(types)))
            ax.set_xticklabels(types, rotation=45, ha='right')
            ax.set_ylabel('Count')
            ax.set_title(f'Event Statistics (Frame {frame_idx+1}/{len(self.all_events)})')
        
        # 添加进度信息
        progress = (frame_idx + 1) / len(self.all_events) * 100
        ax.text(0.02, 0.98, f'Progress: {progress:.1f}%', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python detailed_log_replayer.py <log_directory>")
        sys.exit(1)
    
    log_dir = sys.argv[1]
    replayer = DetailedGameLogReplayer(log_dir)
    replayer.generate_streaming_video()