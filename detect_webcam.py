import argparse
import cv2
from ultralytics import YOLO


def main(weights: str, device: str = "0", conf: float = 0.25, imgsz: int = 960):
    print(f"[INFO] Carregando modelo: {weights}")
    model = YOLO(weights)

    # Mostra todas as classes do modelo (pra conferirmos)
    print("[INFO] Classes do modelo:")
    for idx, name in model.names.items():
        print(f"  {idx}: {name}")

    # Queremos focar só em Helmet e Glasses
    target_names = ["Helmet", "Glasses"]

    # mapear nome -> id (case-insensitive)
    name_to_id = {name.lower(): idx for idx, name in model.names.items()}
    target_ids = []
    for n in target_names:
        if n.lower() in name_to_id:
            target_ids.append(name_to_id[n.lower()])
        else:
            print(f"[WARN] Classe '{n}' não encontrada no modelo!")

    if not target_ids:
        print("[ERRO] Nenhuma das classes alvo foi encontrada. Encerrando.")
        return

    # nomes oficiais (ex: se estiver "helmet" no modelo, usamos isso)
    target_labels = [model.names[i] for i in target_ids]
    print(f"[INFO] Filtrando apenas estas classes: {target_labels} (ids={target_ids})")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERRO] Não foi possível abrir a webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            save=False,
            device=device,
            conf=conf,
            imgsz=imgsz,
            verbose=False
        )

        det = results[0]
        boxes = det.boxes

        presentes = set()

        for box in boxes:
            cls_id = int(box.cls[0].item())
            if cls_id not in target_ids:
                # ignora tudo que não for Helmet ou Glasses
                continue

            label = model.names[cls_id]
            conf_box = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            presentes.add(label)

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"{label} {conf_box:.2f}",
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # Agora o resumo EPI na parte de cima
        faltando = []

        # usamos os labels reais que vieram do modelo
        helmet_label = next((l for l in target_labels if "helmet" in l.lower()), None)
        glasses_label = next((l for l in target_labels if "glass" in l.lower()), None)

        if helmet_label:
            if helmet_label not in presentes:
                faltando.append("Helmet")

        if glasses_label:
            if glasses_label not in presentes:
                faltando.append("Glasses")

        if not faltando:
            status_text = "EPI OK (Helmet + Glasses)"
            color = (0, 255, 0)
        else:
            status_text = "FALTANDO: " + ", ".join(faltando)
            color = (0, 0, 255)

        cv2.putText(
            frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("PPE SH17 - Helmet & Glasses only", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="caminho do .pt treinado")
    parser.add_argument("--device", type=str, default="0", help="0, 1, cpu, etc.")
    parser.add_argument("--conf", type=float, default=0.25, help="confiança mínima")
    parser.add_argument("--imgsz", type=int, default=960, help="tamanho da imagem (lado)")
    args = parser.parse_args()

    main(args.weights, args.device, args.conf, args.imgsz)
