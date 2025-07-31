from utils import get_logger
from rag.vector_store import VectorStoreOperations

class Retriever:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.vector_store = VectorStoreOperations.get_instance()

    def retrieve(self, query: str, top_k: int = 10):
        """
        Truy vấn vector store để lấy các tài liệu liên quan đến truy vấn.

        Args:
            query (str): Chuỗi truy vấn.
            top_k (int): Số lượng tài liệu cần lấy.

        Returns:
            List[dict]: Danh sách các tài liệu được truy vấn.
        """
        try:
            results = self.vector_store.search(query_text=query, top_k=top_k)
            return results
        except Exception as e:
            self.logger.error(f"Lỗi khi truy vấn vector store: {str(e)}")
            raise

if __name__ == "__main__":
    import json

    # Initialize retriever
    retriever = Retriever()

    # Query
    query = "Ngành AI là ngành như thế nào?"

    # Run retrieve và lưu kết quả
    output_file = "search_results.json"
    try:
        results = retriever.retrieve(query, top_k=10)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([result.dict() for result in results], f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error during retrieval: {str(e)}")
        raise