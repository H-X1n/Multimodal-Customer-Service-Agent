import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['PYTHONIOENCODING'] = 'utf-8'

from agent.schemas import UserInput
from agent.services.customer_support_agent import CustomerSupportAgent

agent_lock = threading.Lock()
print_lock = threading.Lock()
progress_lock = threading.Lock()
completed_count = 0
total_count = 0
results_list = []
output_path_global = r'd:\PythonProject\agent项目\资料\submission_result.csv'


def load_questions(csv_path: str) -> list[dict]:
    questions = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append({
                'id': row['id'],
                'question': row['question']
            })
    return questions


def save_results_incremental():
    sorted_results = sorted(results_list, key=lambda x: int(x['id']))
    with open(output_path_global, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'ret'])
        writer.writeheader()
        writer.writerows(sorted_results)


def process_question(agent: CustomerSupportAgent, q: dict) -> dict:
    global completed_count, results_list
    q_id = q['id']
    question = q['question']
    session_id = f"batch_test_{q_id}"
    
    try:
        user_input = UserInput.from_api(
            question=question,
            session_id=session_id,
            stream=False,
        )
        
        with agent_lock:
            response = agent.answer(user_input)
        
        if hasattr(response, 'answer'):
            answer = response.answer
        else:
            answer = str(response)
        
        answer = answer.replace('\n', ' ').replace('\r', ' ')
        
        result = {
            'id': q_id,
            'ret': answer
        }
        
        with progress_lock:
            results_list.append(result)
            completed_count += 1
            pct = completed_count / total_count * 100
            print(f"[{completed_count}/{total_count} ({pct:.1f}%)] ID: {q_id} 完成")
            if completed_count % 10 == 0:
                save_results_incremental()
                print(f"  -> 已保存进度到文件")
        
        return result
        
    except Exception as e:
        result = {
            'id': q_id,
            'ret': f"处理失败: {str(e)}"
        }
        with progress_lock:
            results_list.append(result)
            completed_count += 1
            pct = completed_count / total_count * 100
            print(f"[{completed_count}/{total_count} ({pct:.1f}%)] ID: {q_id} 错误: {e}")
        return result


def main():
    global total_count
    agent = CustomerSupportAgent()
    
    questions_path = r'd:\PythonProject\agent项目\资料\question_public.csv'
    
    questions = load_questions(questions_path)
    total_count = len(questions)
    print(f"共加载 {total_count} 个问题")
    
    results = []
    max_workers = 5
    
    print(f"使用 {max_workers} 个线程并行处理...")
    print(f"结果将增量保存到: {output_path_global}")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_question, agent, q): q
            for q in questions
        }
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    elapsed = time.time() - start_time
    save_results_incremental()
    print(f"\n完成! 共处理 {len(results)} 个问题, 耗时 {elapsed:.1f} 秒")
    print(f"结果已保存到: {output_path_global}")


if __name__ == '__main__':
    main()
