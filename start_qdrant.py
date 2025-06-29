#!/usr/bin/env python3
"""
Script để start Qdrant server bằng Docker cho RAG system.
"""

import subprocess
import time
import sys
import requests


def check_docker_installed():
    """Kiểm tra Docker có cài đặt không."""
    try:
        result = subprocess.run(
            ["docker", "--version"], capture_output=True, text=True, check=True
        )
        print(f"✅ Docker found: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker not found. Please install Docker first.")
        return False


def check_qdrant_running():
    """Kiểm tra Qdrant server có đang chạy không."""
    try:
        response = requests.get("http://localhost:6333/collections", timeout=5)
        if response.status_code == 200:
            print("✅ Qdrant server is already running!")
            return True
    except:
        pass
    return False


def start_qdrant_server():
    """Start Qdrant server với Docker."""
    print("🚀 Starting Qdrant server với Docker...")

    try:
        # Stop existing container if any
        print("🛑 Stopping existing Qdrant containers...")
        subprocess.run(["docker", "stop", "qdrant"], capture_output=True, check=False)
        subprocess.run(["docker", "rm", "qdrant"], capture_output=True, check=False)

        # Start new container
        print("🐳 Starting new Qdrant container...")
        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            "qdrant",
            "-p",
            "6333:6333",
            "-p",
            "6334:6334",
            "qdrant/qdrant",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        container_id = result.stdout.strip()
        print(f"✅ Qdrant container started: {container_id[:12]}")

        # Wait for server to be ready
        print("⏳ Waiting for Qdrant to be ready...")
        for i in range(30):  # Wait up to 30 seconds
            try:
                response = requests.get("http://localhost:6333/collections", timeout=2)
                if response.status_code == 200:
                    print("🎉 Qdrant server is ready!")
                    return True
            except:
                pass

            time.sleep(1)
            print(f"   Waiting... ({i+1}/30)")

        print("❌ Qdrant server didn't start within 30 seconds")
        return False

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Qdrant: {e}")
        print(f"Error output: {e.stderr}")
        return False


def show_qdrant_info():
    """Hiển thị thông tin Qdrant server."""
    try:
        response = requests.get("http://localhost:6333/collections", timeout=5)
        collections = response.json()

        print(f"\n📊 QDRANT SERVER INFO:")
        print(f"   URL: http://localhost:6333")
        print(f"   Collections: {len(collections.get('result', []))}")

        if collections.get("result", []):
            for collection in collections["result"]:
                print(f"   📄 Collection: {collection['name']}")

        print(f"\n🌐 Web UI: http://localhost:6333/dashboard")
        print(f"🔗 API docs: http://localhost:6333/docs")

    except Exception as e:
        print(f"❌ Could not get Qdrant info: {e}")


def main():
    """Main function."""
    print("🇻🇳" + "=" * 50 + "🇻🇳")
    print("🔧  QDRANT SERVER SETUP  🔧")
    print("🇻🇳" + "=" * 50 + "🇻🇳")
    print()

    # Check Docker
    if not check_docker_installed():
        sys.exit(1)

    # Check if already running
    if check_qdrant_running():
        show_qdrant_info()
        return

    # Start Qdrant
    if start_qdrant_server():
        show_qdrant_info()
        print("\n✅ Qdrant setup completed successfully!")
        print("💡 Use 'docker stop qdrant' to stop the server when done.")
    else:
        print("\n❌ Failed to start Qdrant server.")
        print("🔍 Try running manually:")
        print("   docker run -p 6333:6333 qdrant/qdrant")
        sys.exit(1)


if __name__ == "__main__":
    main()
