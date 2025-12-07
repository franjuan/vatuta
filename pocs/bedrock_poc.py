"""
PoC (Proof of Concept) para LangChain 1.0 con AWS Bedrock
Este módulo demuestra cómo usar LangChain 1.0 para interactuar con modelos de AWS Bedrock
"""

import os
from typing import Optional

import boto3
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate


class BedrockPoC:
    """
    Clase para demostrar el uso de LangChain 1.0 con AWS Bedrock
    """

    def __init__(self, profile_name: Optional[str] = None, region: Optional[str] = None):
        """
        Inicializa el cliente de Bedrock con el perfil de AWS especificado

        Args:
            profile_name: Nombre del perfil de AWS a usar (por defecto desde AWS_PROFILE)
            region: Región de AWS donde está disponible Bedrock (por defecto desde AWS_REGION)
        """
        # Obtener perfil desde variable de entorno o usar el proporcionado
        self.profile_name = profile_name or os.getenv("AWS_PROFILE", "IAAdmin")
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self.session = None
        self.bedrock_client = None
        self.llm = None

    def setup_aws_session(self) -> bool:
        """
        Configura la sesión de AWS con el perfil especificado

        Returns:
            bool: True si la configuración fue exitosa, False en caso contrario
        """
        try:
            # Crear sesión con el perfil específico
            self.session = boto3.Session(profile_name=self.profile_name)

            # Crear cliente de Bedrock
            self.bedrock_client = self.session.client("bedrock-runtime", region_name=self.region)

            print(f"✅ Sesión AWS configurada con perfil: {self.profile_name}")
            print(f"✅ Región: {self.region}")

            return True

        except Exception as e:
            print(f"❌ Error configurando sesión AWS: {e}")
            return False

    def setup_langchain_bedrock(self, model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0") -> bool:
        """
        Configura el modelo de LangChain con Bedrock

        Args:
            model_id: ID del modelo de Bedrock a usar
            use_inference_profile: Si usar inference profile en lugar de on-demand

        Returns:
            bool: True si la configuración fue exitosa, False en caso contrario
        """
        try:
            if not self.session:
                print("❌ Sesión AWS no configurada. Ejecuta setup_aws_session() primero.")
                return False

            # Configurar parámetros del modelo
            model_kwargs = {"temperature": 0.7, "max_tokens": 1000}

            # Crear el modelo de LangChain con Bedrock
            self.llm = ChatBedrock(model_id=model_id, client=self.bedrock_client, model_kwargs=model_kwargs)

            print(f"✅ Modelo LangChain configurado: {model_id}")
            return True

        except Exception as e:
            print(f"❌ Error configurando modelo LangChain: {e}")
            return False

    def test_basic_chat(self, message: str = "Hola, ¿cómo estás?") -> Optional[str]:
        """
        Prueba básica de chat con el modelo

        Args:
            message: Mensaje a enviar al modelo

        Returns:
            str: Respuesta del modelo o None si hay error
        """
        try:
            if not self.llm:
                print("❌ Modelo no configurado. Ejecuta setup_langchain_bedrock() primero.")
                return None

            print(f"🤖 Enviando mensaje: {message}")

            # Crear mensaje humano
            human_message = HumanMessage(content=message)

            # Invocar el modelo
            response = self.llm.invoke([human_message])

            print(f"✅ Respuesta recibida: {response.content}")
            return response.content

        except Exception as e:
            print(f"❌ Error en chat básico: {e}")
            return None

    def test_system_prompt(self, system_prompt: str, user_message: str) -> Optional[str]:
        """
        Prueba con prompt del sistema

        Args:
            system_prompt: Prompt del sistema
            user_message: Mensaje del usuario

        Returns:
            str: Respuesta del modelo o None si hay error
        """
        try:
            if not self.llm:
                print("❌ Modelo no configurado. Ejecuta setup_langchain_bedrock() primero.")
                return None

            print(f"🤖 Sistema: {system_prompt}")
            print(f"🤖 Usuario: {user_message}")

            # Crear mensajes del sistema y usuario
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]

            # Invocar el modelo
            response = self.llm.invoke(messages)

            print(f"✅ Respuesta: {response.content}")
            return response.content

        except Exception as e:
            print(f"❌ Error en chat con sistema: {e}")
            return None

    def test_prompt_template(self, template: str, **kwargs) -> Optional[str]:
        """
        Prueba con template de prompt

        Args:
            template: Template del prompt
            **kwargs: Variables para el template

        Returns:
            str: Respuesta del modelo o None si hay error
        """
        try:
            if not self.llm:
                print("❌ Modelo no configurado. Ejecuta setup_langchain_bedrock() primero.")
                return None

            # Crear template de prompt
            prompt = ChatPromptTemplate.from_template(template)

            # Formatear el prompt con las variables
            formatted_prompt = prompt.format(**kwargs)

            print(f"🤖 Prompt formateado: {formatted_prompt}")

            # Crear mensaje
            human_message = HumanMessage(content=formatted_prompt)

            # Invocar el modelo
            response = self.llm.invoke([human_message])

            print(f"✅ Respuesta: {response.content}")
            return response.content

        except Exception as e:
            print(f"❌ Error en template de prompt: {e}")
            return None

    def list_available_models(self) -> Optional[list]:
        """
        Lista los modelos disponibles en Bedrock

        Returns:
            list: Lista de modelos disponibles o None si hay error
        """
        try:
            if not self.bedrock_client:
                print("❌ Cliente Bedrock no configurado.")
                return None

            # Listar modelos disponibles
            response = self.bedrock_client.list_foundation_models()
            models = response.get("modelSummaries", [])

            print("📋 Modelos disponibles en Bedrock:")
            for model in models:
                print(f"  - {model['modelId']} ({model['providerName']})")

            return models

        except Exception as e:
            print(f"❌ Error listando modelos: {e}")
            return None

    def list_inference_profiles(self) -> Optional[list]:
        """
        Lista los inference profiles disponibles en Bedrock

        Returns:
            list: Lista de inference profiles disponibles o None si hay error
        """
        try:
            if not self.bedrock_client:
                print("❌ Cliente Bedrock no configurado.")
                return None

            # Listar inference profiles disponibles
            response = self.bedrock_client.list_inference_profiles()
            profiles = response.get("inferenceProfiles", [])

            print("📋 Inference Profiles disponibles en Bedrock:")
            for profile in profiles:
                print(f"  - {profile['inferenceProfileName']} (ARN: {profile['inferenceProfileArn']})")
                if "modelConfigs" in profile:
                    for model_config in profile["modelConfigs"]:
                        print(f"    └─ Modelo: {model_config.get('modelId', 'N/A')}")

            return profiles

        except Exception as e:
            print(f"❌ Error listando inference profiles: {e}")
            return None


def main():
    """
    Función principal para ejecutar el PoC
    """
    print("🚀 Iniciando PoC de LangChain 1.0 con AWS Bedrock")
    print("=" * 60)

    # Crear instancia del PoC (usará variables de entorno por defecto)
    poc = BedrockPoC()

    # Configurar sesión AWS
    if not poc.setup_aws_session():
        print("❌ No se pudo configurar la sesión AWS. Verifica tu configuración.")
        return

    # Listar modelos disponibles
    print("\n📋 Listando modelos disponibles...")
    poc.list_available_models()

    # Listar inference profiles disponibles
    print("\n📋 Listando inference profiles disponibles...")
    poc.list_inference_profiles()

    # Configurar modelo LangChain con inference profile
    print("\n🔧 Configurando modelo LangChain con inference profile...")
    if not poc.setup_langchain_bedrock(model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0"):
        print("❌ No se pudo configurar el modelo LangChain.")
        return

    # Prueba 1: Chat básico
    print("\n🧪 Prueba 1: Chat básico")
    print("-" * 30)
    poc.test_basic_chat("Hola, ¿puedes explicarme qué es LangChain?")

    # Prueba 2: Chat con sistema
    print("\n🧪 Prueba 2: Chat con prompt del sistema")
    print("-" * 30)
    poc.test_system_prompt(
        "Eres un asistente experto en programación Python. Responde de manera técnica y precisa.",
        "¿Cuáles son las mejores prácticas para manejar excepciones en Python?",
    )

    # Prueba 3: Template de prompt
    print("\n🧪 Prueba 3: Template de prompt")
    print("-" * 30)
    poc.test_prompt_template(
        "Eres un {role}. Explica {topic} de manera {style}.",
        role="experto en AWS",
        topic="qué es Amazon Bedrock",
        style="clara y concisa",
    )

    print("\n✅ PoC completado exitosamente!")


if __name__ == "__main__":
    main()
