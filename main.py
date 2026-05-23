from modes.live_mode import run_live
from modes.upload_mode import run_upload

def main():
    print("Patient Posture Monitoring System")
    print("1 - Live Camera Mode")
    print("2 - Upload Image/Video Mode")
    
    choice = input("Select mode (1 or 2): ").strip()
    
    if choice == "1":
        run_live()
    elif choice == "2":
        path = input("Enter file path: ").strip()
        run_upload(path)
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()