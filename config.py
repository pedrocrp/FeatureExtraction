from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QFormLayout, QMessageBox,
    QGroupBox, QCheckBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QEvent
from components.Buttons.Button import Button
import requests
from components.MainBoard.MainBoard import MainBoard
from dotenv import set_key, load_dotenv
from utils.get_path import get_resource_path
from utils.api_manager import get_next_count_id
from utils.get_path import get_resource_path
import os
from pymodbus.client import ModbusTcpClient
from dotenv import set_key, load_dotenv
from pathlib import Path

COIL_TRIGGER_ADDRESS = 2
COIL_RETRY_ATTEMPTS = 3   # Número de tentativas
COIL_RETRY_DELAY = 0.5 
ENV_FILE = Path(__file__).parent.parent.parent / ".env"
MODBUS_IP = "192.168.3.5"
MODBUS_PORT = 502
TEST_COIL_ADDRESS = 2  # Endereço do coil que queremos testar

class ConfigPage(QWidget):
    peso_amostra_atualizado = pyqtSignal(float)
    latest_weight=0
    #peso_bag = 0
    numero_amostra_recebido = pyqtSignal(str)

    #self.main_board = MainBoard()

    API_BASE_URL = "http://localhost:8001"

    MODBUS_COUNT = 4

    def __init__(self, parent=None, navigate_fn=None):
        super().__init__(parent)
        self.navigate_to = navigate_fn
        self.setWindowTitle("Configurações")
        self.setGeometry(100, 100, 800, 600)
        self.contador_amostra = 0
        self.em_edicao = False
        self.campos_editados = set()
        self.invalid_style = "border: 1px solid red;"
        self.valid_style = ""

        
        layout = QVBoxLayout()
        self.setLayout(layout)

        button_layout = QHBoxLayout()

        self.reset_falha_button = Button(
            text="Resetar Falhas",
            style_class="button-lightgray-black",
            clicked=self.reset_falhas,
            font_size=19,
            size=(380, 70),
        )
        button_layout.addWidget(self.reset_falha_button)

        layout.addLayout(button_layout)

        amostra_layout = QHBoxLayout()
        label_amostra = QLabel("Número da Amostra:")
        self.numero_amostra_input = QLineEdit()
        self.numero_amostra_input.setReadOnly(True)
        self.numero_amostra_input.setFixedWidth(200)
        amostra_layout.addWidget(label_amostra)
        amostra_layout.addWidget(self.numero_amostra_input)
        amostra_layout.addStretch()
        layout.addLayout(amostra_layout)

        sent_group = QGroupBox("Configurações Enviadas para o CLP (Leitura/Escrita)")
        sent_layout = QFormLayout()
        labels = [
            "Tempo maximo sem Produto (ms):",
            "Tempo para Desligar Contador (ms):",
            "Quantidade de Sementes no Bag:",
            "Quantidade de Peneiras Selecionadas:"
        ]
        self.sent_inputs = []

        for i in range(self.MODBUS_COUNT):
            input_field = QLineEdit()
            input_field.setFixedWidth(200)
            input_field.setPlaceholderText("Vazio")
            label_text = labels[i] if i < len(labels) else f"Endereço {i}"
            sent_layout.addRow(label_text, input_field)
            self.sent_inputs.append(input_field)

        #self.peso_bag_input = QLineEdit()
        #self.peso_bag_input.setFixedWidth(200)
        #self.peso_bag_input.textChanged.connect(self.atualizar_peso_bag)
        #self.peso_bag_input.setPlaceholderText("Vazio")
        #sent_layout.addRow("Peso da Bag (kg):", self.peso_bag_input)

        sent_group.setLayout(sent_layout)
        layout.addWidget(sent_group)

        received_group = QGroupBox("Dados Recebidos do CLP (Somente Leitura)")
        received_layout = QFormLayout()
        self.received_inputs = []
        input_field = QLineEdit()
        input_field.setFixedWidth(200)
        input_field.setReadOnly(True)
        input_field.setPlaceholderText("Peso atual")
        received_layout.addRow("Peso da Amostra (g):", input_field)
        self.received_inputs.append(input_field)
        for i in range(1, self.MODBUS_COUNT):
            input_field = QLineEdit()
            input_field.setVisible(False)
            self.received_inputs.append(input_field)
        received_group.setLayout(received_layout)
        layout.addWidget(received_group)

        self.atualizar_button = Button(
            text="Atualizar Valores do CLP",
            style_class="button-lightgray-black",
            clicked=self.atualizar_valores,
            font_size=16,
            size=(380, 60),
        )
        self.save_button = Button(
            text="Salvar Configurações no CLP",
            style_class="button-lightgray-black",
            clicked=self.salvar_configuracoes,
            font_size=16,
            size=(380, 60),
        )

        self.cancel_button = Button(
            text="Cancelar Edição",
            style_class="button-lightgray-black",
            clicked=self.cancelar_edicao,
            font_size=16,
            size=(380, 60),
        )

        action_buttons_layout = QHBoxLayout()
        action_buttons_layout.addWidget(self.cancel_button)        
        action_buttons_layout.addWidget(self.atualizar_button)
        action_buttons_layout.addWidget(self.save_button)
        layout.addLayout(action_buttons_layout)
        
        load_dotenv(get_resource_path("../.env"))  # Carrega variáveis do .env
        self.update_sample_number_from_env()  # Atualiza número ao iniciar

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.atualizar_dados_recebidos)
        self.timer.start(1000)

        for i, input_field in enumerate(self.sent_inputs):
            input_field.installEventFilter(self)
            input_field.setProperty("field_index", i)  # Para identificar o campo no eventFilter
        
        # Conecte o evento de mudança de texto para validação em tempo real
        for input_field in self.sent_inputs:
            input_field.textChanged.connect(self.validar_campo_em_tempo_real)
        #self.peso_bag_input.textChanged.connect(self.validar_campo_em_tempo_real)
        self.numero_amostra_recebido.connect(self.atualizar_numero_amostra)
    
    def eventFilter(self, obj, event):
        # Detecta quando um campo ganha ou perde foco
        if event.type() == QEvent.FocusIn:
            self.em_edicao = True
            index = obj.property("field_index")
            self.campos_editados.add(index)
        elif event.type() == QEvent.FocusOut:
            self.em_edicao = False
            
        return super().eventFilter(obj, event)


    def atualizar_dados_recebidos(self):
        try:
            # 1. Obter peso da balança
            try:
                response_balanca = requests.get(f"{self.API_BASE_URL}/clp/balanca", timeout=2)
                response_balanca.raise_for_status()
                peso_data = response_balanca.json()
                
                if 'peso' in peso_data:
                    peso = peso_data['peso']
                    self.__save_weight_on_coil_trigger(peso)

                    if len(self.received_inputs) > 0:
                        self.received_inputs[0].setText(f"{peso:.2f}")
                        self.peso_amostra_atualizado.emit(float(peso))
                        self.latest_weight=float(peso)
                else:
                    error_msg = peso_data.get('error', 'Formato inválido')
                    print(f"Erro na balança: {error_msg}")
                    if len(self.received_inputs) > 0:
                        self.received_inputs[0].setText(f"Erro: {error_msg}")
            except Exception as e:
                print(f"Erro ao obter peso: {e}")
                if len(self.received_inputs) > 0:
                    self.received_inputs[0].setText("Erro conexão")
    
            # 2. Obter outros registros
            try:
                response_holding = requests.get(f"{self.API_BASE_URL}/clp/holding", timeout=2)
                response_holding.raise_for_status()
                holding = response_holding.json().get("registers", [])
                
                for i in range(min(len(holding), self.MODBUS_COUNT)):
                    if (not self.sent_inputs[i].hasFocus() and
                        i not in self.campos_editados):
                        self.sent_inputs[i].setText(str(holding[i]) if holding[i] != 0 else "")
            except Exception as e:
                print(f"Erro ao obter holding registers: {e}")
                # Limpa os campos em caso de erro
                for i in range(self.MODBUS_COUNT):
                    if not self.sent_inputs[i].hasFocus():
                        self.sent_inputs[i].clear()
    
        except Exception as e:
            print(f"Erro geral em atualizar_dados_recebidos: {e}")
            # Limpa os campos em caso de erro geral
            for i in range(self.MODBUS_COUNT):
                if not self.sent_inputs[i].hasFocus():
                    self.sent_inputs[i].clear()

    def reset_falhas(self):
        try:
            requests.post(f"{self.API_BASE_URL}/clp/reset-falhas")
            QMessageBox.information(self, "Resetar Falhas", "Falhas do sistema resetadas com sucesso!")
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao resetar falhas: {e}")

    def update_sample_number_from_env(self):
        """Atualiza o número da amostra baseado no .env"""
        try:
            current_count = int(os.getenv("CURRENT_COUNT_NUMBER", "0"))
            next_num = current_count + 1
            self.numero_amostra_input.setText(str(next_num))
        except Exception as e:
            print(f"Erro ao atualizar número da amostra: {e}")
            self.numero_amostra_input.setText("N/A")

    def iniciar_lote(self):
        # 1. Primeiro faz a validação dos campos
        if not self.validar_campos():
            return  # Não continua se a validação falhar

        # 2. Confirmação com o usuário
        reply = QMessageBox.question(
            self,
            "Confirmar Início",
            "Deseja realmente iniciar o novo lote?\n\n"
            f"Peso da Amostra: {self.received_inputs[0].text()}g\n",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return  # Usuário cancelou

        try:
            # 3. Simula Envio de comando para o CLP
            #data = {"values": [False]}  # Envia True para o coil 0 (iniciar lote)
            #response = requests.post(f"{self.API_BASE_URL}/clp/coils", json=data, timeout=2)
            #response.raise_for_status()  # Lança exceção se status não for 2xx

            # 4. Atualiza o .env com o peso da bag e reseta a contagem
            #set_key(get_resource_path("../.env"), "BAG_WEIGHT", str(self.peso_bag))
            set_key(get_resource_path("../.env"), "CURRENT_COUNT_NUMBER", "0")
            self.update_sample_number_from_db() 

            # 5. Atualiza o número da amostra exibido (agora será 0 + 1 = 1)
            self.update_sample_number_from_env()

            # 6. Navega para a tela de operação
            if hasattr(self, 'navigate_to') and callable(self.navigate_to):
                msg_box = QMessageBox()
                msg_box.setWindowTitle("Iniciar Lote")
                msg_box.setText("Processo de lote iniciado com sucesso!\n\nAguarde, você será redirecionado...")
                msg_box.setIcon(QMessageBox.Information)
                msg_box.setStandardButtons(QMessageBox.Ok)

                QTimer.singleShot(2000, lambda: self.navigate_to("Operation"))
                msg_box.exec_()
            else:
                QMessageBox.warning(self, "Aviso", "Função de navegação não disponível")

        except requests.exceptions.RequestException as e:
            QMessageBox.critical(
                self, 
                "Erro de Comunicação", 
                f"Falha ao comunicar com o CLP:\n{str(e)}"
            )
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Erro Inesperado", 
                f"Ocorreu um erro inesperado:\n{str(e)}"
            )

    def atualizar_valores(self):
        try:
            self.atualizar_dados_recebidos()
            QMessageBox.information(self, "Atualizar", "Valores atualizados com sucesso!")
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao atualizar valores: {e}")
    
    #def atualizar_peso_bag(self):
    #    try:
    #        self.peso_bag = int(self.peso_bag_input.text())
    #    except ValueError:
    #        # Mostra mensagem de erro ao usuário
    #        QMessageBox.warning(
    #            self,
    #            "Valor inválido",
    #            "Por favor, insira um valor numérico válido para o peso da bag.",
    #            QMessageBox.Ok
    #        )
    #        # Mantém o foco no campo e seleciona o texto para fácil correção
    #        self.peso_bag_input.setFocus()
    #        self.peso_bag_input.selectAll()
    #        # Define um valor padrão ou mantém o anterior
    #        self.peso_bag = 0  # Ou não altera self.peso_bag se quiser manter o último valor válido

    def atualizar_numero_amostra(self):
        """Atualiza o número da amostra exibido com o próximo ID disponível"""
        batch_id = self.get_current_batch_id()
        next_id = get_next_count_id(batch_id=batch_id)
        if next_id:
            self.numero_amostra_input.setText(str(next_id))
            self.current_count_id = next_id
        else:
            self.numero_amostra_input.setText("N/A")
            self.current_count_id = None

    def salvar_configuracoes(self):
        try:
            values = []
            for i in range(self.MODBUS_COUNT):
                value = self.sent_inputs[i].text()
                if value:
                    values.append(int(value))
                else:
                    values.append(0)
            requests.post(f"{self.API_BASE_URL}/clp/registers", json={"values": values})

            #requests.post(f"{self.API_BASE_URL}/clp/coils", json={"values": coils})

            self.em_edicao = False

            QMessageBox.information(self, "Salvar Configurações", "Configurações salvas com sucesso!")
        except ValueError:
            QMessageBox.warning(self, "Erro", "Por favor, insira valores válidos em todos os campos.")
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao salvar configurações: {e}")

    def validar_campos(self):
        """Valida se todos os campos obrigatórios estão preenchidos corretamente"""
        errors = []
        field_labels = [
            "Tempo máximo sem Produto (ms)",
            "Tempo para Desligar Contador (ms)",
            "Quantidade de Sementes no Bag",
            "Quantidade de Peneiras Selecionadas"
        ]
        
        # Verifica campos de configuração do CLP
        for i, (field, label) in enumerate(zip(self.sent_inputs, field_labels)):
            if not field.text().strip():
                errors.append(f"{label} não pode estar vazio")
            elif not field.text().strip().isdigit():
                errors.append(f"{label} deve ser um número inteiro")
        
        # Verifica peso da bag
        #if not self.peso_bag_input.text().strip():
        #    errors.append("Peso da Bag não pode estar vazio")
        #else:
        #    try:
        #        float(self.peso_bag_input.text())
        #    except ValueError:
        #        errors.append("Peso da Bag deve ser um número válido")
        
        # Verifica peso da amostra
        peso_amostra = self.received_inputs[0].text()
        if not peso_amostra or "Erro" in peso_amostra:
            errors.append("Peso da Amostra - verifique a conexão com a balança")
        
        if errors:
            QMessageBox.warning(
                self,
                "Campos Inválidos",
                "Por favor, corrija os seguintes campos:\n\n• " + "\n• ".join(errors)
            )
            return False
        
        return True

    def validar_campo_em_tempo_real(self):
        """Validação visual em tempo real"""
        sender = self.sender()
        if not sender.text().strip():
            sender.setStyleSheet(self.invalid_style)
        else:
            sender.setStyleSheet(self.valid_style)
    
    def get_current_batch_id(self):
        """Retorna o ID do lote atualmente selecionado"""
        # Implementação depende de como você armazena o lote selecionado
        # Exemplo 1: Se estiver no main_board
        if hasattr(self, 'main_board') and hasattr(self.main_board, 'selected_batch_id'):
            return self.main_board.selected_batch_id

    def cancelar_edicao(self):
        self.em_edicao = False
        self.campos_editados.clear()
        self.atualizar_dados_recebidos()  # Restaura os valores originais
        QMessageBox.information(self, "Edição Cancelada", "As alterações não salvas foram descartadas.")

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()

    def update_sample_number_from_db(self):
        """Versão final com tratamento completo de erros"""
        try:
            if not hasattr(self, 'numero_amostra_input'):
                return
    
            # Tentativa 1: Obter do banco via API
            try:
                if hasattr(self.parent(), 'operation_page'):
                    batch_id = self.parent().operation_page.get_current_batch_id()
                    if batch_id:
                        response = requests.get(
                            f"http://localhost:8000/api/seed-counts/next-id/?batch_id={batch_id}",
                            timeout=1
                        )
                        if response.status_code == 200:
                            data = response.json()
                            next_num = data['batch_count_number']
                            self.numero_amostra_input.setText(str(next_num))
                            set_key(get_resource_path("../.env"), "CURRENT_COUNT_NUMBER", str(next_num - 1))
                            return
            except requests.exceptions.RequestException:
                pass  # Vamos para o fallback

            # Tentativa 2: Obter do .env
            try:
                current_count = int(os.getenv("CURRENT_COUNT_NUMBER", "0"))
                self.numero_amostra_input.setText(str(current_count + 1))
            except:
                self.numero_amostra_input.setText("N/A")

        except Exception as e:
            print(f"Erro crítico ao atualizar número: {e}")
            self.numero_amostra_input.setText("Erro")

    def __save_weight_on_coil_trigger(self, peso):
        '''
        Salva o valor da contagem atual garantindo o estado do coil
        '''
        try:
            # 1. Garante que o coil está ativado
            #if not self.ensure_coil_state(True):
            #    raise Exception("Não foi possível ativar o coil de trigger")

            # 2. Salva o peso no .env
            set_key(ENV_FILE, "LATEST_WEIGHT", str(peso))
            print(f"[DEBUG] Peso {peso} salvo com coil ativo")

        except Exception as e:
            error_msg = f"Falha ao salvar peso com trigger: {str(e)}"
            print(f"[ERRO] {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)

    def ensure_coil_state(self, target_state=True):
        """
        Garante que o coil alvo esteja no estado desejado
        Retorna True se bem sucedido, False caso contrário
        """
        client = None
        try:
            client = ModbusTcpClient(MODBUS_IP, port=MODBUS_PORT)

            for attempt in range(COIL_RETRY_ATTEMPTS):
                try:
                    # 1. Conectar ao CLP
                    if not client.connect():
                        raise Exception("Falha na conexão com o CLP")

                    # 2. Ler o estado atual
                    read_response = client.read_coils(COIL_TRIGGER_ADDRESS, count=1)
                    if read_response.isError():
                        raise Exception(f"Erro na leitura: {read_response}")

                    current_state = read_response.bits[0]

                    # 3. Se já está no estado desejado, retorna sucesso
                    if current_state == target_state:
                        return True

                    # 4. Tentar escrever o estado desejado
                    write_response = client.write_coil(COIL_TRIGGER_ADDRESS, target_state)
                    if write_response.isError():
                        raise Exception(f"Erro na escrita: {write_response}")

                    # 5. Verificar se a escrita foi efetiva
                    verify_response = client.read_coils(COIL_TRIGGER_ADDRESS, count=1)
                    if verify_response.isError():
                        raise Exception(f"Erro na verificação: {verify_response}")

                    if verify_response.bits[0] == target_state:
                        return True

                except Exception as e:
                    print(f"Tentativa {attempt + 1} falhou: {str(e)}")
                    if attempt == COIL_RETRY_ATTEMPTS - 1:  # Última tentativa
                        raise

                    time.sleep(COIL_RETRY_DELAY)

        finally:
            if client:
                client.close()

        return False