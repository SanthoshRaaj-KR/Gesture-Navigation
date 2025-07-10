# === main.py ===
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

class GestureController:
    def __init__(self):
        self.last_gesture = None
        self.last_action_time = 0
        self.action_delay = 0.7  
        self.browser_window = None
        self.gesture_queue = queue.Queue()
        
    def find_browser_window(self):
        """Find and return the first available browser window"""
        try:
            windows = gw.getAllWindows()
            browser_names = ["Brave", "Chrome", "Firefox", "Edge", "Safari", "Opera"]
            
            for window in windows:
                if window.title.strip() != "":
                    for browser_name in browser_names:
                        if browser_name.lower() in window.title.lower():
                            print(f"[🌐 BROWSER] Found: {window.title}")
                            return window
            return None
        except Exception as e:
            print(f"[⚠️ BROWSER ERROR] {e}")
            return None
    
    def execute_gesture_action(self, gesture):
        """Execute the browser action based on gesture"""
        print(f"[🎯 ACTION] Executing: {gesture}")
        
        # Find browser window
        self.browser_window = self.find_browser_window()
        if not self.browser_window:
            print("[❌ ERROR] No browser window found!")
            return False
        
        try:
            if gesture == "restore_browser":
                print("[📂 RESTORE] Restoring browser window...")
                if self.browser_window.isMinimized:
                    self.browser_window.restore()
                    time.sleep(0.5)  # Wait for window to restore
                self.browser_window.activate()
                
            elif gesture == "minimize_browser":
                print("[📉 MINIMIZE] Minimizing browser window...")
                self.browser_window.minimize()
                
            elif gesture == "tab_left":
                print("[⬅️ TAB LEFT] Switching to previous tab...")
                self.browser_window.activate()
                time.sleep(0.2)
                pyautogui.hotkey('ctrl', 'shift', 'tab')
                
            elif gesture == "tab_right":
                print("[➡️ TAB RIGHT] Switching to next tab...")
                self.browser_window.activate()
                time.sleep(0.2)
                pyautogui.hotkey('ctrl', 'tab')
                
            else:
                print(f"[❓ UNKNOWN] Unknown gesture: {gesture}")
                return False
            
            # Add 1-second delay after performing action
            print("[⏱️ WAITING] Action completed, waiting 1 second...")
            time.sleep(1.0)
            return True
            
        except Exception as e:
            print(f"[❌ ACTION ERROR] {e}")
            return False
    
    def process_gestures(self):
        """Process gestures from the queue"""
        print("[🎮 CONTROLLER] Gesture processor started")
        
        while True:
            try:
                if not self.gesture_queue.empty():
                    # Get the latest gesture (clear queue to avoid backlog)
                    latest_gesture = None
                    while not self.gesture_queue.empty():
                        latest_gesture = self.gesture_queue.get_nowait()
                    
                    if latest_gesture:
                        current_time = time.time()
                        
                        # Check if enough time has passed and gesture is different
                        if (current_time - self.last_action_time >= self.action_delay and 
                            latest_gesture != self.last_gesture):
                            
                            print(f"[🆕 NEW GESTURE] {latest_gesture}")
                            
                            # Execute the action
                            if self.execute_gesture_action(latest_gesture):
                                self.last_gesture = latest_gesture
                                self.last_action_time = current_time
                                print(f"[✅ SUCCESS] {latest_gesture} completed")
                            else:
                                print(f"[❌ FAILED] {latest_gesture} failed")
                        else:
                            if latest_gesture == self.last_gesture:
                                print(f"[🔄 REPEAT] Ignoring repeated gesture: {latest_gesture}")
                            else:
                                time_left = self.action_delay - (current_time - self.last_action_time)
                                print(f"[⏳ COOLDOWN] Wait {time_left:.1f}s before next action")
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[⚠️ PROCESSOR ERROR] {e}")
                time.sleep(0.5)
    
    def start(self):
        """Start the gesture control system"""
        print("[🚀 STARTUP] Starting Gesture Control System...")
        print("[ℹ️ INFO] Supported gestures: restore_browser, minimize_browser, tab_left, tab_right")
        
        try:
            # Start gesture processor in separate thread
            processor_thread = threading.Thread(
                target=self.process_gestures,
                daemon=True
            )
            processor_thread.start()
            print("[✅ SUCCESS] Gesture processor started")
            
            # Start gesture prediction (this will block)
            print("[📷 CAMERA] Starting camera for gesture detection...")
            print("[🤲 READY] Show your gestures to control the browser!")
            start_prediction(self.gesture_queue)
            
        except KeyboardInterrupt:
            print("\n[👋 SHUTDOWN] Gracefully shutting down...")
        except Exception as e:
            print(f"[❌ FATAL ERROR] {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[🔚 END] Gesture control system terminated")

def main():
    """Main function"""
    print("=" * 60)
    print("🖐 GESTURE BROWSER CONTROLLER")
    print("=" * 60)
    print("Controls:")
    print("  - restore_browser: Restore/activate browser window")
    print("  - minimize_browser: Minimize browser window")
    print("  - tab_left: Switch to previous tab")
    print("  - tab_right: Switch to next tab")
    print("  - Press 'q' in camera window to quit")
    print("=" * 60)
    
    controller = GestureController()
    controller.start()

if __name__ == "__main__":
    main()