# === gesture_actions.py ===
import pyautogui
import pygetwindow as gw
import time
import queue
import threading
import sys
import os

# Add error handling for imports
try:
    from gesture_predictor import start_prediction
except ImportError as e:
    print(f"[ERROR] Failed to import gesture_predictor: {e}")
    sys.exit(1)

pyautogui.FAILSAFE = False
COOLDOWN_TIME = 2  # seconds

def perform_action(label):
    print(f"[DEBUG] Performing action for label: {label}")
    try:
        # More flexible window detection
        windows = gw.getAllWindows()
        browser_window = None
        
        # Look for browser windows (Brave, Chrome, Firefox, etc.)
        browser_names = ["Brave", "Chrome", "Firefox", "Edge", "Safari"]
        
        for window in windows:
            if window.title.strip() != "":  # Skip empty titles
                for browser_name in browser_names:
                    if browser_name.lower() in window.title.lower():
                        browser_window = window
                        break
                if browser_window:
                    break

        if not browser_window:
            print("[⚠️ NO BROWSER] No browser window found.")
            return

        print(f"[DEBUG] Found browser window: {browser_window.title}")

        if label == "restore_browser":
            if browser_window.isMinimized:
                browser_window.restore()
                time.sleep(0.3)
            browser_window.activate()
            
        elif label == "minimize_browser":
            browser_window.minimize()
            
        elif label == "tab_left":
            browser_window.activate()
            time.sleep(0.1)
            pyautogui.hotkey('ctrl', 'shift', 'tab')
            
        elif label == "tab_right":
            browser_window.activate()
            time.sleep(0.1)
            pyautogui.hotkey('ctrl', 'tab')
            
        print(f"[✅ SUCCESS] Action '{label}' completed")
        
    except Exception as e:
        print(f"[⚠️ ACTION ERROR] {e}")
        import traceback
        traceback.print_exc()

def action_listener(pred_queue):
    """Listen for predictions and execute actions"""
    last_label = None
    last_action_time = 0
    
    print("[DEBUG] Action listener started")
    
    while True:
        try:
            if not pred_queue.empty():
                label = pred_queue.get_nowait()  # Non-blocking get
                current_time = time.time()
                
                print(f"[DEBUG] Received label: {label}")
                
                # Add cooldown to prevent rapid repeated actions
                if (label != last_label or 
                    current_time - last_action_time > COOLDOWN_TIME):
                    
                    print(f"[🤲 Gesture] Performing: {label}")
                    perform_action(label)
                    last_label = label
                    last_action_time = current_time
                else:
                    print(f"[DEBUG] Skipping repeated action: {label}")
                    
            time.sleep(0.1)  # Small delay to prevent busy waiting
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[⚠️ LISTENER ERROR] {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("[🚀 STARTUP] Starting gesture control system...")
    
    try:
        # Create queue for communication
        prediction_queue = queue.Queue()
        
        # Start action listener in daemon thread
        action_thread = threading.Thread(
            target=action_listener, 
            args=(prediction_queue,), 
            daemon=True
        )
        action_thread.start()
        
        print("[✅ SUCCESS] Action listener started")
        
        # Start prediction (this will block)
        print("[📷 CAMERA] Starting gesture prediction...")
        start_prediction(prediction_queue)
        
    except KeyboardInterrupt:
        print("\n[👋 SHUTDOWN] Gracefully shutting down...")
    except Exception as e:
        print(f"[❌ FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[🔚 END] Program terminated")