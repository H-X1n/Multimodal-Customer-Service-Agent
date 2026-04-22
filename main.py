from __future__ import annotations

import json

from agent.schemas import ContentBlock, UserInput
from agent.services.customer_support_agent import CustomerSupportAgent


def main() -> None:
    agent = CustomerSupportAgent()
    print("多模态客服智能体已启动，输入 exit 退出。")
    while True:
        query = input("\n用户问题: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        image_raw = input("图片路径（多个用英文逗号分隔，可留空）: ").strip()
        image_paths = [item.strip() for item in image_raw.split(",") if item.strip()]

        contents = [ContentBlock(type="text", data=query)]
        for img_path in image_paths:
            contents.append(ContentBlock(type="image", data=img_path))

        response = agent.answer(UserInput(contents=contents))
        print("\n助手回复:")
        print(response.answer)
        if response.referenced_images:
            print("\n相关图片:")
            print(json.dumps(response.referenced_images, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
