#!/usr/bin/env python3
"""
FOOLPROOF DATA PROCESSOR
Tries multiple industry-standard approaches until one works
"""
import os
import sys
import json
import subprocess

def run_method(script_name, method_name):
    """Run a processing method and return success/failure"""
    print(f"\n{'='*50}")
    print(f"TRYING: {method_name}")
    print(f"{'='*50}")
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_name], 
                               capture_output=True, text=True, timeout=300)  # 5 min timeout
        
        if result.returncode == 0:
            print(f"✅ {method_name} SUCCESS!")
            print("Output:", result.stdout[-500:])  # Last 500 chars
            return True, result.stdout
        else:
            print(f"❌ {method_name} failed with return code {result.returncode}")
            print("Error:", result.stderr[-500:])
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {method_name} timed out (>5 minutes)")
        return False, "Timeout"
        
    except Exception as e:
        print(f"💥 {method_name} crashed: {e}")
        return False, str(e)

def main():
    """Try each method until one works"""
    
    print("🚀 FOOLPROOF DATA PROCESSING")
    print("Will try multiple approaches until one succeeds...")
    
    # Check if data exists
    required_files = ['data/train.parquet', 'data/test.parquet']
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Missing: {file}")
            return
    
    # Create submissions folder
    os.makedirs('submissions', exist_ok=True)
    
    # Define methods to try (in order of reliability)
    methods = [
        ("chunked_analysis.py", "Chunked Processing (Most Reliable)"),
        ("dask_analysis.py", "Dask Framework (Industry Standard)"),
        ("sqlite_converter.py", "SQLite Database (Bulletproof)"),
    ]
    
    results = {
        "attempted_methods": [],
        "successful_method": None,
        "submission_files": [],
        "errors": []
    }
    
    # Try each method
    for script, description in methods:
        if not os.path.exists(script):
            print(f"⚠️ {script} not found, skipping...")
            continue
            
        success, output = run_method(script, description)
        
        method_result = {
            "method": description,
            "script": script,
            "success": success,
            "output_snippet": output[-200:] if output else ""
        }
        
        results["attempted_methods"].append(method_result)
        
        if success:
            results["successful_method"] = description
            
            # Check for submission files
            submission_files = []
            for file in os.listdir('submissions'):
                if file.endswith('.parquet'):
                    submission_files.append(f"submissions/{file}")
            
            results["submission_files"] = submission_files
            break
        else:
            results["errors"].append(f"{description}: {output}")
    
    # Final results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    if results["successful_method"]:
        print(f"✅ SUCCESS: {results['successful_method']}")
        print(f"\n📁 Submission files created:")
        for file in results["submission_files"]:
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"   - {file} ({size_mb:.1f} MB)")
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Upload any submission file to Kaggle")
        print(f"2. Check your score on the leaderboard") 
        print(f"3. We can improve from there!")
        
    else:
        print("❌ ALL METHODS FAILED")
        print("\nErrors encountered:")
        for error in results["errors"][-3:]:  # Show last 3 errors
            print(f"   - {error}")
        
        print(f"\nThis is unusual - please share the error messages")
    
    # Save detailed results
    with open('processing_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Detailed results saved to: processing_results.json")
    
    return results["successful_method"] is not None

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)