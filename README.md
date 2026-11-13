# FashionTrendIntelligence
AI services requests - Openclassrooms AI Engineer Project 1

## Segmentation de vêtements

Le script `segment_clothes.py` envoie des images au modèle
[`sayeed99/segformer_b3_clothes`](https://huggingface.co/sayeed99/segformer_b3_clothes)
via l'API d'inférence Hugging Face et affiche l'image originale, le masque de
classes et leur superposition.

Installez les dépendances du projet avec `uv sync`, puis définissez un token
Hugging Face autorisé à utiliser les Inference Providers :

```bash
export HF_TOKEN="hf_..."
```

Vous pouvez aussi enregistrer `HF_TOKEN=hf_...` dans un fichier `.env` à la
racine du projet. Lancez le script avec un ou plusieurs fichiers ou dossiers :

```bash
uv run python segment_clothes.py asset/img/image_0.png
uv run python segment_clothes.py asset/img/
```

Formats acceptés : JPG, PNG et WEBP.

## API serverless sur AWS Lambda

`POST /segment` reçoit une image JPEG, PNG ou WebP de 4 Mio maximum dans le
corps binaire de la requête. Il renvoie le tableau JSON de Hugging Face
contenant un `label` et un `mask` PNG encodé en base64 par classe détectée.
La réponse peut aussi contenir un `score` fourni par Hugging Face.

### Déploiement

Dans AWS Secrets Manager, créez un secret dans la région du déploiement avec
un champ JSON `HF_TOKEN`. Le token Hugging Face doit avoir la permission
**Make calls to Inference Providers**. Avec AWS SAM CLI et des identifiants AWS
configurés, lancez ces commandes depuis la racine du dépôt :

```bash
sam validate --lint
sam deploy --guided
```

Indiquez le nom du secret pour le paramètre `HuggingFaceSecret`. Relevez
l'URL `ApiUrl` dans les sorties de la pile CloudFormation. Le rôle de
déploiement doit pouvoir lire le secret.

### Appel de l'API

Le client signe la requête avec AWS Signature Version 4. L'identité AWS utilisée
doit disposer des permissions `lambda:InvokeFunctionUrl` et
`lambda:InvokeFunction` sur la fonction. Avec AWS CLI v2 et un profil autorisé :

```bash
eval "$(aws configure export-credentials --format env)"
API_URL="https://<id>.lambda-url.<region>.on.aws/"
AWS_REGION="<region>"
IMAGE="asset/img/image_0.png"
PAYLOAD_SHA256="$(sha256sum "$IMAGE" | cut -d ' ' -f 1)"
SESSION_HEADER=()
if [[ -n "${AWS_SESSION_TOKEN:-}" ]]; then
  SESSION_HEADER=(-H "x-amz-security-token: $AWS_SESSION_TOKEN")
fi

curl --fail-with-body \
  --aws-sigv4 "aws:amz:$AWS_REGION:lambda" \
  --user "$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY" \
  "${SESSION_HEADER[@]}" \
  -H "x-amz-content-sha256: $PAYLOAD_SHA256" \
  -H 'Content-Type: image/png' \
  --data-binary "@$IMAGE" \
  "${API_URL%/}/segment"
```
