# Работа с Inference Services


[[_TOC_]]
В **KubeFlow** развёртывание моделей осуществляется с помощью компонента **Inference Service**, который является **CRD (Custom Resource Definition)**. **Inference Service** Обеспечивает масштабируемость и доступ к сервисам извне.

## 1. Подготовка к развёртыванию

Чтобы развернуть  **Inference Service**, необходимо сначала создать хранилище, где будут лежать модели и конфигурации их запуска. В качестве внутреннего сервиса для запуска моделей будет использоваться **TorchServe**.

**yaml**-файл для создания хранилища выглядит следующим образом:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: torchserve-claim
  namespace: kubeflow-megaputer
spec:
  storageClassName: nfs-client
  resources:
    requests:
      storage: 50Gi
  accessModes:
    - ReadWriteMany
```

Здесь мы создаём **PersistentVolumeClaim** с именем **torchserve-claim** в пространстве имён **kubeflow-megaputer**. В качестве класса хранилища (**storageClassName**) указывается **nfs-client**, в котором выделяются 50 ГБ пространства (storage: 50Gi) в режиме множественного доступа (**ReadWriteMany**).

В качестве типа хранилища может использоваться и **microk8s-hostpath**, создаваемый **MicroK8S** по-умолчанию, но использование **NFS**-хранилища более удобно, т.к. для пользователя это обычная сетевая папка.

Создать развернуть **PersistentVolumeClaim** можно командой:

```bash
microk8s kubectl apply -f pvc-torchserve.yaml
```

В **jupyter**-блокноте команда выглядит так:

```bash
!kubectl apply -f pvc-torchserve.yaml
```

**Примечание:** В **jupyter**-блокноте не нужно указывать `microk8s`, т.к. сам блокнот развёрнут в среде **MicroK8S**. Однако перед командой ставится знак `!`.

После развёртывания  **PersistentVolumeClaim**, в директории **nfs-client**'а, указанной в Разделе 2.1 докукумента **install_kubeflow_v1.5** (в нашем случае `/mnt/nfs`)

Проверить наличие директории, созданной **PersistentVolumeClaim** командой:

```bash
ls -l /mnt/nfs
```

Вывод должен быть похож на следующий:

```bash
drwxrwxrwx 6 root   root    4096 авг 21 16:22 kubeflow-megaputer-torchserve-claim-pvc-0aceda3b-1fb6-4f35-9629-2c3cd2c00a51
```

где `0aceda3b-1fb6-4f35-9629-2c3cd2c00a51` - UID, полученный во время создания **PersistentVolumeClaim**.

**Внимание:** UID, полученный во время создания **PersistentVolumeClaim** генерируется автоматически и каждый раз разный!

В созданную директорию ` kubeflow-megaputer-torchserve-claim-pvc-xxx` следует помещать модели для развёртывания.

Модели для развёртывания представляют с собой два файла : конфигурационный **config.properties**  и файл самой модели - **mar**-файл, расположенные в директоиях **config** и **model-store** соттветственно. Ниже представлен вывод команды `ls -LR /mmt/nfs/kubeflow-megaputer-torchserve-claim-pvc-0aceda3b-1fb6-4f35-9629-2c3cd2c00a51/nllb-200-1dot3b`:

```bash
drwxrwxr-x 2 rtx4090 rtx4090 4096 окт 14  2022 config
drwxrwxr-x 2 rtx4090 rtx4090 4096 апр 12 07:55 model-store

./config:
итого 12
-rw-rw-r-- 1 rtx4090 rtx4090 644 окт 14  2022  config.properties

./model-store:
итого 3143164
-rw-rw-r-- 1 rtx4090 rtx4090 3218592192 апр 12 07:55 nllb-200-1dot3b.mar
```

## 2. Создание InferenceService

Создать **Inference Service** можно двумя способами: с помощью **yaml**-файла или посредством **python**-библиотеки.

### 2.1. Создание InferenceService с помощью yaml-файла

**yaml**-файл для создания **Inference Service** выглядит следующим образом:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: "nllb-200-1dot3b"
  namespace: kubeflow-megaputer
spec:
  predictor:
    pytorch:
      protocolVersion: v1
      runtimeVersion: 0.5.3-gpu
      image: docker.io/yurkoff/torchserve-cuda-11.3-kfs:0.5.3-gpu
      storageUri: pvc://torchserve-claim/nllb-200-1dot3b
      resources:
        requests:
          cpu: "2"
          memory: 8Gi
          nvidia.com/gpu: "1"
        limits:
          cpu: "2"
          memory: 12Gi
          nvidia.com/gpu: "1"
    minReplicas: 1
    maxReplicas: 1
    timeout: 300
```

В поле **name** задаётся имя сервиса, поле **image** - образ, на основе которого будет запускаться сервис (в нашем случае **TorchServe**). В поле **storageUri** указывается хранилище и путь относительно него, где расположены файлы с конфигурацией запуска модели (в нашем случае **config.properties** для **TorchServe**) и сама модель (для **TorchServe** это **mar**-файл). Далее следует описание необходимых и максимально требуемых ресурсов для запуска **Inference Service**; тут следует отметить требование к наличию доступного ресурса в виде **GPU**: **nvidia.com/gpu**. В конце указывается минимальное и максимальное количество реплик и таймаут. Количество реплик влияет на то, какое количество моделей будет развёрнуто в сервисе. Таймаут указывает на максимально ожидаемое время ответа от модели (время inference). По истечении таймаута соединение сбрасывается.

**Внимание:** Файлы модели должны располагаться в директории `kubeflow-megaputer-torchserve-claim-pvc-xxx/nllb-200-1dot3b`, где `xxx` - UID, полученный в Разделе 1 во время создания **PersistentVolumeClaim**.

### 2.2. Создание InferenceService с помощью **python**-библиотеки

Скачать требуюмую версию CUDA с официального сайта Nvidia https://developer.nvidia.com/cuda-toolkit-archive.

**Внимание**: Для видеокарты **Nvidia Tesla K80** последняя поддерживаемая версия **CUDA 11.4.4**!

Для получения ссылки на скачивание необходимо выбрать опрационную систему. Самый простой способ скачать файл для локальной установки:

```python
from kubernetes import client
from kubernetes.client import V1ResourceRequirements

from kserve import KServeClient
from kserve import constants
from kserve import utils
from kserve import V1beta1PredictorSpec
from kserve import V1beta1TorchServeSpec
from kserve import V1beta1InferenceServiceSpec
from kserve import V1beta1InferenceService

kserve_client = KServeClient()

pytorch_predictor_nllb_large=V1beta1PredictorSpec(
    pytorch=V1beta1TorchServeSpec(
        runtime_version='0.5.3-gpu',
        image="docker.io/yurkoff/torchserve-cuda-11.3-kfs:0.5.3-gpu",
        storage_uri='pvc://torchserve-claim/nllb-200-1dot3b',
        resources=V1ResourceRequirements(
            requests={'cpu':'2000m', 'memory':'8Gi', 'nvidia.com/gpu': '1'},
            limits={'cpu':'2000m', 'memory':'12Gi', 'nvidia.com/gpu': '1'}
        ),
    ),
    min_replicas=1,
    max_replicas=1,
    timeout=120,
)


isvc_nllb_large = V1beta1InferenceService(api_version=constants.KSERVE_V1BETA1,
                                          kind=constants.KSERVE_KIND,
                                          metadata=client.V1ObjectMeta(name='nllb-200-1dot3b',
                                          namespace=namespace),
                                          spec=V1beta1InferenceServiceSpec(predictor=pytorch_predictor_nllb_large))

kserve_client.create(isvc_nllb_large)
```

Для начала импортируются необходимые библиотеки и создаётся объект **KServeClient**. Далее идёт описание предиктора (**V1beta1PredictorSpec**), в роли которого выступает **TorchServe** (**V1beta1TorchServeSpec**). Его параметры аналогичны параметрам, описанным в **yaml**-файле. После создаётся объект **V1beta1InferenceService**, в который передаётся только что созданный предиктор. В заключении создаётся только что описанный **Inference Service** с помощью метода **create**.

**Внимание:** Файлы модели должны располагаться в директории `kubeflow-megaputer-torchserve-claim-pvc-xxx/nllb-200-1dot3b`, где `xxx` - UID, полученный в Разделе 1 во время создания **PersistentVolumeClaim**.

## 3. Мониторинг и управление Inference Service

Все команды управления **Inference Service** будут показаны для командной строки. Для адаптации их к блокноту, нужно в начале команды убрать `microk8s`и поставить знак `!`, как это было указано в Разделе 1.

Чтобы получить список **Infernce Service**, нужно выполнить команду:

```bash
microk8s kubectl get inferenceservices --all-namespaces
```

Здесь мы указываем, что хотим получить список всех сервисов во всех пространствах имён (**namespace**). По-умолчанию все сервисы создаются в пространстве имён `kubeflow-megaputer`. Вывод будет выглядеть следующим образом:

```bash
NAMESPACE            NAME              URL                                                     READY   PREV   LATEST   PREVROLLEDOUTREVISION   LATESTREADYREVISION                       AGE
kubeflow-megaputer   nllb-200-1dot3b   http://nllb-200-1dot3b.kubeflow-megaputer.example.com   True           100                              nllb-200-1dot3b-predictor-default-00001   63s
```

Тут следует обратить внимание на флаг `READY`, который должен быть в значении `True`.



Для просмотра запущенных **POD**'ов используется команда:

```bash
microk8s kubectl get pods -n namespace
```

где в качестве `namespace` указывается пространство имён, в котором должны располагаться  **POD**'ы, например,`kubeflow-megaputer`:

```bash
microk8s kubectl get pods -n kubeflow-megaputer
```

Полезно создать переменную NAMESPACE, в которую записать требуемое имя пространства имён:

```bash
NAMESPACE=kubeflow-megaputer
```

Тогда переписать команду можно так:

```bash
microk8s kubectl get pods -n $NAMESPACE
```

**Примечание:** В **jupyter**-блокноте переменные заключаются в фигурные скобки и пришутся без знака `$`:

```bash
!kubectl get pods -n {NAMESPACE}
```

Вывод команды будет следующим:

```
NAME                                                              READY   STATUS    RESTARTS   AGE
ml-pipeline-visualizationserver-7c8dfd5cb-w8gmv                   2/2     Running   8          225d
ml-pipeline-ui-artifact-8dcf69986-5jrmh                           2/2     Running   8          225d
megaputer-notebook-0                                              2/2     Running   8          225d
nllb-200-1dot3b-predictor-default-00001-deployment-5887b5495xx9   3/3     Running   0          39s
```

Наш **POD** имеет имя `nllb-200-1dot3b-predictor-default-00001-deployment-5887b5495xx9`.
**Внимание:** Если  **POD** находится в состоянии **Pending** или  **Init:CrashLoopBackOff**, то сервис не может запуститься. При этом необходимо проверить правильность выполнения предыдущих шагов, а также, что файлы моделей расположены в правильных директориях!

Часто бывает полезным записать имя **POD**'а в переменную. Делается это командой:

```bash
POD_NLLB_LARGE=$(microk8s kubectl get pods -n $NAMESPACE | grep -Eo "(nllb-200-1dot3b[_A-Za-z0-9-]+)")
```

где `nllb-200-1dot3b` имя  **Infernce Service**, т.к.  с него и начинается имя **POD**'а. Переменная  `POD_NLLB_LARGE` может содержать не одно значение, а список **POD**'ов, удовлетворяющих заданному условию. Дело в том, что при развёртывании нескольких реплик, каждая реплика создаёт свой **POD**, начинающийся с имени сервиса и заканчивающийся уникальным UUID (в нашем примере это `5887b5495xx9`). Тогда обращение к конкретному **POD**'у осуществляется с указанием его номера в списке в квадратных скобках (нумерация начинается с нуля), например: `$POD_NLLB_LARGE[0]`.

Зная имя **POD**'а можно посмотреть логи контейнеров, расположенных в нём:

```ba
microk8s kubectl logs -n $NAMESPACE $POD_NLLB_LARGE -c storage-initializer
microk8s kubectl logs -n $NAMESPACE $POD_NLLB_LARGE -c kserve-container --tail 50
```

**POD**, создаваемый **Infernce Service**, содержит два контейнера: `storage-initializer` и `kserve-container`. Первый нужен для   монтирования модели и её конфигурации запуска внутрь **POD**'а, а второй это непосредственно образ  **TorchServe**, указанный при создании сервиса. Пример вывода первой команды:

```bash
/usr/local/lib/python3.7/site-packages/ray/autoscaler/_private/cli_logger.py:61: FutureWarning: Not all Ray CLI dependencies were found. In Ray 1.4+, the Ray CLI, autoscaler, and dashboard will only be usable via `pip install 'ray[default]'`. Please update your install command.
  "update your install command.", FutureWarning)
[I 230609 15:56:48 initializer-entrypoint:13] Initializing, args: src_uri [/mnt/pvc/nllb-200-1dot3b] dest_path[ [/mnt/models]
[I 230609 15:56:48 storage:52] Copying contents of /mnt/pvc/nllb-200-1dot3b to local
[I 230609 15:56:48 storage:263] Linking: /mnt/pvc/nllb-200-1dot3b/model-store to /mnt/models/model-store
[I 230609 15:56:48 storage:263] Linking: /mnt/pvc/nllb-200-1dot3b/config to /mnt/models/config

```

Вывод второй команды лучше ограничивать некоторым количеством строк (флаг `--tail`), т.к. она содержит лог **TorchServe**. Пример лога:

```bash
[I 230609 15:56:56 __main__:81] Wrapper : Model names ['nllb-200-1dot3b'], inference address http//0.0.0.0:8085, management address http://0.0.0.0:8081, model store /mnt/models/model-store
[I 230609 15:56:56 TorchserveModel:59] kfmodel Predict URL set to 0.0.0.0:8085
[I 230609 15:56:56 TorchserveModel:61] kfmodel Explain URL set to 0.0.0.0:8085
[I 230609 15:56:56 TSModelRepository:36] TSModelRepo is initialized
[I 230609 15:56:56 kfserver:150] Registering model: nllb-200-1dot3b
[I 230609 15:56:56 kfserver:120] Setting asyncio max_workers as 6
[I 230609 15:56:56 kfserver:127] Listening on port 8080
[I 230609 15:56:56 kfserver:129] Will fork 1 workers
2023-06-09T15:58:22,560 [DEBUG] main org.pytorch.serve.wlm.ModelVersionedRefs - Adding new version 1.0 for model nllb-200-1dot3b
2023-06-09T15:58:22,560 [DEBUG] main org.pytorch.serve.wlm.ModelVersionedRefs - Setting default version to 1.0 for model nllb-200-1dot3b
2023-06-09T15:58:39,098 [DEBUG] main org.pytorch.serve.wlm.ModelVersionedRefs - Setting default version to 1.0 for model nllb-200-1dot3b
2023-06-09T15:58:39,098 [INFO ] main org.pytorch.serve.wlm.ModelManager - Model nllb-200-1dot3b loaded.
2023-06-09T15:58:39,099 [DEBUG] main org.pytorch.serve.wlm.ModelManager - updateModel: nllb-200-1dot3b, count: 1
2023-06-09T15:58:39,108 [DEBUG] W-9000-nllb-200-1dot3b_1.0 org.pytorch.serve.wlm.WorkerLifeCycle - Worker cmdline: [/home/venv/bin/python3.7, /home/venv/lib/python3.7/site-packages/ts/model_service_worker.py, --sock-type, unix, --sock-name, /home/model-server/tmp/.ts.sock.9000]
2023-06-09T15:58:39,112 [INFO ] main org.pytorch.serve.ModelServer - Initialize Inference server with: EpollServerSocketChannel.
2023-06-09T15:58:39,190 [INFO ] main org.pytorch.serve.ModelServer - Inference API bind to: http://0.0.0.0:8085
2023-06-09T15:58:39,190 [INFO ] main org.pytorch.serve.ModelServer - Initialize Management server with: EpollServerSocketChannel.
2023-06-09T15:58:39,191 [INFO ] main org.pytorch.serve.ModelServer - Management API bind to: http://0.0.0.0:8081
2023-06-09T15:58:39,191 [INFO ] main org.pytorch.serve.ModelServer - Initialize Metrics server with: EpollServerSocketChannel.
2023-06-09T15:58:39,192 [INFO ] main org.pytorch.serve.ModelServer - Metrics API bind to: http://0.0.0.0:8082
Model server started.
2023-06-09T15:58:39,466 [WARN ] pool-3-thread-1 org.pytorch.serve.metrics.MetricCollector - worker pid is not available yet.
2023-06-09T15:58:40,280 [INFO ] pool-3-thread-1 TS_METRICS - CPUUtilization.Percent:0.0|#Level:Host|#hostname:nllb-200-1dot3b-predictor-default-00001-deployment-5887b5495xx9,timestamp:1686326320
2023-06-09T15:58:40,281 [INFO ] pool-3-thread-1 TS_METRICS - DiskAvailable.Gigabytes:294.68211364746094|#Level:Host|#hostname:nllb-200-1dot3b-predictor-default-00001-deployment-5887b5495xx9,timestamp:1686326320
2023-06-09T15:58:40,281 [INFO ] pool-3-thread-1 TS_METRICS - DiskUsage.Gigabytes:138.92472076416016|#Level:Host|#hostname:nllb-200-1dot3b-predictor-default-00001-deployment-5887b5495xx9,timestamp:1686326320
2023-06-09T15:58:40,281 [INFO ] pool-3-thread-1 TS_METRICS - DiskUtilization.Percent:32.0|#Level:Host|#hostname:nllb-200-1dot3b-predictor-default-00001-deployment-5887b5495xx9,timestamp:1686326320
2023-06-09T15:58:40,281 [INFO ] pool-3-thread-1 TS_METRICS - GPUMemoryUtilization.Percent:0.07731119791666667|#Level:Host,device_id:0|#hostname:nllb-200-1dot3b-predictor-default-00001-deployment-5887b5495xx9,timestamp:1686326320
2023-06-09T15:58:40,281 [INFO ] pool-3-thread-1 TS_METRICS - GPUMemoryUsed.Megabytes:19|#Level:Host,device_id:0|#hostname:nllb-200-1dot3b-predictor-default-00001-deployment-5887b5495xx9,timestamp:1686326320
2023-06-09T15:58:40,281 [INFO ] pool-3-thread-1 TS_METRICS - GPUUtilization.Percent:0|#Level:Host,device_id:0|#hostname:nllb-200-1dot3b-predictor-default-00001-deployment-5887b5495xx9,timestamp:1686326320
2023-06-09T15:58:40,281 [INFO ] pool-3-thread-1 TS_METRICS - MemoryAvailable.Megabytes:22652.5234375|#Level:Host|#hostname:nllb-200-1dot3b-predictor-default-00001-deployment-5887b5495xx9,timestamp:1686326320
2023-06-09T15:58:40,281 [INFO ] pool-3-thread-1 TS_METRICS - MemoryUsed.Megabytes:9318.21484375|#Level:Host|#hostname:nllb-200-1dot3b-predictor-default-00001-deployment-5887b5495xx9,timestamp:1686326320
2023-06-09T15:58:40,281 [INFO ] pool-3-thread-1 TS_METRICS - MemoryUtilization.Percent:29.3|#Level:Host|#hostname:nllb-200-1dot3b-predictor-default-00001-deployment-5887b5495xx9,timestamp:1686326320
2023-06-09T15:58:40,852 [INFO ] W-9000-nllb-200-1dot3b_1.0-stdout MODEL_LOG - Listening on port: /home/model-server/tmp/.ts.sock.9000
2023-06-09T15:58:40,852 [INFO ] W-9000-nllb-200-1dot3b_1.0-stdout MODEL_LOG - [PID]58
2023-06-09T15:58:40,852 [INFO ] W-9000-nllb-200-1dot3b_1.0-stdout MODEL_LOG - Torch worker started.
2023-06-09T15:58:40,852 [INFO ] W-9000-nllb-200-1dot3b_1.0-stdout MODEL_LOG - Python runtime: 3.7.5
2023-06-09T15:58:40,852 [DEBUG] W-9000-nllb-200-1dot3b_1.0 org.pytorch.serve.wlm.WorkerThread - W-9000-nllb-200-1dot3b_1.0 State change null -> WORKER_STARTED
2023-06-09T15:58:40,856 [INFO ] W-9000-nllb-200-1dot3b_1.0 org.pytorch.serve.wlm.WorkerThread - Connecting to: /home/model-server/tmp/.ts.sock.9000
2023-06-09T15:58:40,863 [INFO ] W-9000-nllb-200-1dot3b_1.0-stdout MODEL_LOG - Connection accepted: /home/model-server/tmp/.ts.sock.9000.
2023-06-09T15:58:40,865 [INFO ] W-9000-nllb-200-1dot3b_1.0 org.pytorch.serve.wlm.WorkerThread - Flushing req. to backend at: 1686326320865
2023-06-09T15:58:40,909 [INFO ] W-9000-nllb-200-1dot3b_1.0-stdout MODEL_LOG - model_name: nllb-200-1dot3b, batchSize: 1
2023-06-09T15:59:14,248 [INFO ] W-9000-nllb-200-1dot3b_1.0-stdout MODEL_LOG - Model M2M100ForConditionalGeneration from path /home/model-server/tmp/models/e206d7bc07844331ae52e52795484351 loaded successfully
2023-06-09T15:59:14,252 [INFO ] W-9000-nllb-200-1dot3b_1.0-stdout MODEL_LOG - Model started on cuda:0 device
```
Удалить можно с помощью команды:

```bash
microk8s kubectl delete inferenceservice nllb-200-1dot3b -n $NAMESPACE
```

где `nllb-200-1dot3b` имя  **Infernce Service**, котрый мы создавали.

**Примечание:** В **jupyter**-блокноте есть собственная команда для удаления **Infernce Service** через созданный клиент **KServeClient**

```python
 kserve_client.delete('nllb-200-1dot3b', namespace='kubeflow-megaputer')
```

Но я не рекомендую её использовать, т.к. из-за неё вознивает ошибка: **Infernce Service** удаляется, а его имя остаётся отображаться при запросе действующих **Infernce Service**:

```bash
microk8s kubectl get inferenceservices --all-namespaces
```

