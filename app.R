#Detectar librerías usadas en tu script o proyecto
#required_pkgs <- renv::dependencies()$Package

# Reinstalar las librerías detectadas
#for (pkg in required_pkgs) {
#       install.packages(pkg, dependencies = TRUE)
#}



#install.packages("cli")

#remove.packages(cli)
#remove.packages("shiny")
#install.packages("caret", dependencies = TRUE)

library(shiny)
library(rsconnect)
#library(caret)
library(DT)
library(cli)
library(ggplot2)
library(DataExplorer)
library(e1071)

library(caret)
#library(RWeka)  # Para el clasificador C4.5 (J48)
#library(RWekajars)# revisar el error al plot arbol
#library(rJava)

library(caTools) # Para createDataPartition
library(pROC)    # Para curvas ROC y AUC
library(FSelector) # Para filtros como CFS
# Cargar el paquete necesario para gráficos
library(corrplot)
#library(dplyr)
library(partykit)
library(shinyjs) #para habilitar los botones de descargas
# Cargar dataset precargado

#librarias del algoritmo GA
library(GA)

#rsconnect::appDependencies()

library(C50)

#arritmia <- read.csv("arrhythmia.csv", header = FALSE, sep = ",", na.strings = c("?", " "))

ui <- fluidPage(
        useShinyjs(),  # Habilitar shinyjs
        titlePanel(
                div(
                        img(src = "corazon2.jpg", height = "55px", width = "100px"), "Optimización de Modelos para Identificar Arritmia Cardíaca con Aprendizaje Automático",
                        img(src = "meta.jpg", height = "90px", width = "130px")
                        #strong("Optimización de Modelos para Identificar Arritmia Cardíaca con Aprendizaje Automático"),
                       #img(src = "corazon2.jpg", height = "50px", style = "margin-left: 20px;")
                   )
                ),
        sidebarLayout(
                sidebarPanel(
                        # Sidebar general para todas las pestañas excepto "Variables Relevantes (Filtros)"
                        conditionalPanel(
                                #condition = "input.main_tabs != 'Variables Relevantes (Filtros)'",
                                condition = "input.main_tabs != 'Variables Relevantes (Filtros)' && input.main_tabs != 'Variables Relevantes (Metaheurísticas)' && input.main_tabs != 'Referencias'",
                                
                                h3("Menú Principal"),
                                actionButton("reset_app", "Reiniciar Interfaz", icon = icon("redo"), class="btn btn-danger"), #class = "btn-warning"
                                fileInput("file1", "Cargar Dataset", accept = c(".csv", ".xlsx")),
                                actionButton("load_sample", "Dataset Arritmia Precargado", class = "btn-lg btn-success"),
                                checkboxInput("header", "Leer encabezados", TRUE),
                                uiOutput("data_view"),
                                selectInput("target_var", "Seleccionar Variable a Predecir:", choices = NULL),
                                verbatimTextOutput("class_info"),
                                selectInput("negative_class", "Clase Negativa (1):", choices = NULL),
                                selectInput("positive_class", "Clases Positivas (2):", choices = NULL, multiple = TRUE),
                                actionButton("convert_binary", "Convertir a Clase Binaria")
                        ),
                        # Sidebar específico para "Variables Relevantes (Filtros)"
                        conditionalPanel(
                                
                                condition = "input.main_tabs == 'Variables Relevantes (Filtros)'",
                                actionButton("reset_filter", "Reiniciar Filtro y clasificador", icon = icon("redo"), class="btn btn-danger"),
                                
                                h4("Variable Predictora (Clase):"),
                                verbatimTextOutput("selected_target"),
                                
                                
                                h3("Selección de Variables Relevantes"),
                                hr(),
                                selectInput("filter_method", "1. Seleccione un Método Filtro:",
                                            choices = c("CFS", "Consistency", "Chi.squared", "Information Gain",
                                                        "Gain Ratio", "Symmetrical Uncertainty", "OneR")),
                                
                                actionButton("apply_filter", "Aplicar Filtro", class = "btn btn-primary"),
                                verbatimTextOutput("num_selected_vars"),
                                
                                hr(),
                                
                                selectInput("classifier", "2. Seleccione un Clasificador:",
                                            choices = c("C5.0", "Rpart", "PART", "JRip", "OneR", "SVMlin", "KNN", "Random Forest")),
                                actionButton("apply_classifier_30times", "Ejecutar Clasificador 30 Veces",class = "btn btn-primary"),
                                hr(),
                                
                                #selectInput("tree_run", "Seleccione la ejecución para graficar el árbol:", choices = 1:30)
                                
                                
                                #actionButton("apply_filter", "Aplicar Filtro y Clasificador", class = "btn-primary"),
                                #actionButton("reset_filter", "Reiniciar", class = "btn-warning"),
                                
                                
                                
                        ),
                        conditionalPanel(
                                
                                condition = "input.main_tabs == 'Variables Relevantes (Metaheurísticas)'",
                                
                                numericInput("popSize", "Tamaño de la población:", value = 50, min = 10, step = 1),
                                numericInput("pcrossover", "Probabilidad de cruce:", value = 0.8, min = 0, max = 1, step = 0.1),
                                numericInput("pmutation", "Probabilidad de mutación:", value = 0.1, min = 0, max = 1, step = 0.1),
                                numericInput("elitism", "Número de élites:", value = 2, min = 1, step = 1),
                                numericInput("maxiter", "Número máximo de iteraciones:", value = 5, min = 1, step = 1),
                                actionButton("run_ga", "Ejecutar Algoritmo Genético", class = "btn btn-primary"),
                                
                                verbatimTextOutput("num_selected_vars_GA"),
                                
                                hr(),
                                
                                selectInput("classifier_GA", "Seleccione un Clasificador:",
                                            choices = c("C5.0", "Rpart", "PART", "JRip", "OneR", "SVMlin", "KNN", "Random Forest")),
                                actionButton("apply_classifier_30times_GA", "Ejecutar Clasificador 30 Veces",class = "btn btn-primary"),
                                hr(),
                                
                                #selectInput("tree_run_GA", "Seleccione la ejecución para graficar el árbol:", choices = 1:30)
                                
                                
                                # actionButton("reset_filter", "Reiniciar Filtro y clasificador", icon = icon("redo"), class="btn btn-danger"),
                                
                                
                        ),
                        conditionalPanel(
                                
                                condition = "input.main_tabs == 'Referencias'",
                                img(src = "dacyti3.jpg", height = "462px", width = "429px"),
                        ),
                ),
                mainPanel(
                        tabsetPanel(
                                id = "main_tabs",
                                tabPanel("Información",
                                         h4(strong("Bienvenido ", style = "color: blue;" )),
                                         h5(p(strong("Para comenzar, cargue un archivo CSV o seleccione el dataset precargado usando las opciones de la barra lateral.", style = "color: red;" ))),
                                         
                                         p("Una vez cargado el dataset, se habilitarán las pestañas de datos, preprocesamiento y modelado."),
                                         
                                         h4("Acerca de la arritmia cardiaca y el electrocardiograma (ECG)"),
                                         p(
                                                 "Una arritmia es un latido o ritmo cardíaco irregular. Puede clasificarse en distintas clases."
                                         ),                                        
                                         p(
                                                 "La electrocardiografía (ECG) es un método para analizar la condición cardíaca de un paciente. 
                                                 Un ECG es una representación eléctrica de la actividad contráctil del corazón y se puede registrar utilizando electrodos de superficie colocados en las extremidades o el tórax del paciente."
                                                 
                                         ),
                                         h4("Información del conjunto de datos"),
                                         p(
                                                 "Esta base de datos contiene 279 atributos, 206 de los cuales son de valor lineal y el resto son nominales."
                                         ),
                                         p(
                                                 "El objetivo es distinguir entre la presencia y ausencia de arritmia cardiaca y clasificarla en uno de los 16 grupos. 
                                                 La clase 01 se refiere al ECG 'normal', las clases 02 a 15 a diferentes clases de arritmia y la clase 16 al resto de las no clasificadas. 
                                                 Por el momento, existe un programa informático que realiza dicha clasificación."
                                         ),
                                         
                                         #img(src = "arritmia.jpg", style = "display: block; margin-left: auto; margin-right: auto;", height = "432px", width = "800px")
                                         # Imagen dentro del tabPanel
                                         img(src = "arritmia.jpg", height = "380px", width = "780px")
                                         
                                ),
                                tabPanel("Datos",
                                         tabsetPanel(
                                                 tabPanel("Vista de Datos", dataTableOutput("data_table")),
                                                 tabPanel("Resumen", uiOutput("summary_view")),
                                                 tabPanel("Valores Nulos",
                                                          plotOutput("na_plot"),
                                                          textOutput("na_message"),
                                                          actionButton("remove_high_na", "Eliminar Columnas con >80% NAs", class = "btn-danger"),
                                                          h4("Métodos de Imputación"),
                                                          radioButtons("imputation_methods", "Seleccione un Método de Imputación:",
                                                                       choices = c("Media" = "mean", "Mediana" = "median", "Moda" = "mode"),
                                                                       selected = "mean"),
                                                          actionButton("apply_imputation", "Aplicar Imputación", class = "btn-lg btn-success"),
                                                          downloadButton("download_data", "Descargar Dataset Imputado", class = "butt"),
                                                          tags$head(tags$style(".butt{background-color:orange;} .butt{color: black;}")) # background color and font color
                                                 ),
                                                 tabPanel("Modelos de clasificación",
                                                          h3("Modelos de clasificación con el dataset Completo"),
                                                          hr(),
                                                          selectInput("classifier_all", "Seleccione un Clasificador:",
                                                                      choices = c("C5.0", "Rpart", "PART", "JRip", "OneR", "SVMlin", "KNN", "Random Forest")),
                                                          actionButton("apply_classifier_30times_all", "Ejecutar Clasificador 30 Veces",class = "btn btn-primary"),
                                                          #verbatimTextOutput("metrics_summary_all"),
                                                          #DT::dataTableOutput("summary_table_all"),
                                                          DTOutput("metrics_table_all"),
                                                          downloadButton("download_results_all", "Descargar Resultados", class = "butt1"),
                                                          tags$head(tags$style(".butt1{background-color:orange;} .butt{color: black;}"))
                                                          
                                                 )
                                         )
                                ),
                                tabPanel("Variables Relevantes (Filtros)",
                                         tabsetPanel(
                                                 # tabPanel("Datos Filtrados", dataTableOutput("filtered_data_table")),
                                                 # tabPanel("Resumen", verbatimTextOutput("filtered_summary")),
                                                 #tabPanel("Curva ROC", plotOutput("roc_plot")),
                                                 
                                                 # h3("Resumen del Dataset Filtrado"),
                                                 #DT::dataTableOutput("summary_table"),
                                                 
                                                 # h3("Gráfico de Correlación"),
                                                 # plotOutput("correlation_plot"),
                                                 
                                                 #h3("Resumen del Modelo de Clasificación"),
                                                 #verbatimTextOutput("model_summary")
                                                 
                                                 
                                                 tabPanel("Subconjunto Seleccionado",
                                                          h3("Dataset Filtrado"),
                                                          DT::dataTableOutput("summary_table"),
                                                          downloadButton("download_data2", "Descargar Dataset Filtrado", class = "butt1"),
                                                          tags$head(tags$style(".butt1{background-color:orange;} .butt{color: black;}")),
                                                          DTOutput("metrics_table"),
                                                          #verbatimTextOutput("metrics_summary"),
                                                          downloadButton("download_results", "Descargar Resultados", class = "butt1"),
                                                          tags$head(tags$style(".butt1{background-color:orange;} .butt{color: black;}")),
                                                          #hr(),
                                                          #plotOutput("tree_plot"), #era para graficar los 30 arboles
                                                          #downloadButton("download_trees", "Descargar las 30 gráficas de árboles generados", class = "butt1"),
                                                          tags$head(tags$style(".butt1{background-color:orange;} .butt{color: black;}")),
                                                          
                                                          
                                                          
                                                          
                                                          
                                                 ),
                                                 tabPanel("Gráficas y Reglas", 
                                                          selectInput("tree_run", "Seleccione el número de ejecución para mostrar las reglas del arbol:", choices = 1:30),
                                                          
                                                          downloadButton("download_tree_plot", "Descargar Gráfico del Árbol", class = "butt1"),  # Botón de descarga
                                                          tags$head(tags$style(".butt1{background-color:orange;} .butt{color: black;}")),
                                                          
                                                          plotOutput("tree_plot", height = "600px"),  # Gráfico del árbol
                                                          
                                                          
                                                          downloadButton("download_tree_text", "Descargar Árbol como Texto", class = "butt1"),
                                                          tags$head(tags$style(".butt1{background-color:orange;} .butt{color: black;}")),
                                                          
                                                          verbatimTextOutput("tree_text"),  # Salida para el texto del árbol
                                                          
                                                          
                                                 ),
                                                 
                                                 tabPanel("Resultados 30 veces", 
                                                          verbatimTextOutput("result_30"),
                                                          downloadButton("download_summary_30", "Descargar 30 ejecuciones", class = "butt1"),
                                                          tags$head(tags$style(".butt1{background-color:orange;} .butt{color: black;}")),
                                                          DTOutput("metrics_table_30"),
                                                          
                                                          
                                                 ),
                                                 
                                                 tabPanel("Gráfico de Correlación",
                                                          h3("Gráfico de Correlación"),
                                                          plotOutput("correlation_plot")
                                                 ),
                                                 tabPanel("Resumen del subconjunto",
                                                          h3("Resumen del Modelo de Clasificación"),
                                                          verbatimTextOutput("model_summary")
                                                 )
                                                 
                                                 
                                                 
                                                 #,
                                                 #tabPanel("Datos Filtrados", dataTableOutput("filtered_data_table")),
                                                 #tabPanel("Curva ROC", plotOutput("roc_plot"))
                                                 
                                                 
                                                 
                                         )
                                ),
                                tabPanel("Variables Relevantes (Metaheurísticas)",
                                         tabsetPanel(
                                                 tabPanel("Subconjunto Seleccionado",
                                                          h3("selección de datos relevantes por medio de metaheurística"),
                                                          
                                                          DTOutput("selected_vars_table_GA"),
                                                          downloadButton("download_vars_GA", "Descargar Variables Seleccionadas", class = "butt1"),
                                                          tags$head(tags$style(".butt1{background-color:orange;} .butt{color: black;}")),
                                                          
                                                          
                                                          DTOutput("metrics_table_GA"),
                                                          downloadButton("download_results_GA", "Descargar Resultados", class = "butt1"),
                                                          tags$head(tags$style(".butt1{background-color:orange;} .butt{color: black;}")),
                                                          #hr(),
                                                          
                                                          #verbatimTextOutput("metrics_summary"),
                                                          #plotOutput("tree_plot_GA"), #era para graficar los arboles
                                                          #downloadButton("download_trees_GA", "Descargar las 30 gráficas de árboles generados", class = "butt1"),
                                                          #tags$head(tags$style(".butt1{background-color:orange;} .butt{color: black;}")),
                                                          
                                                          #selectInput("tree_run_GA", "Seleccione el número de ejecución para mostrar las reglas del arbol:", choices = 1:30),
                                                          
                                                          #downloadButton("download_tree_text_GA", "Descargar Árbol como Texto", class = "butt1"),
                                                         # tags$head(tags$style(".butt1{background-color:orange;} .butt{color: black;}")),
                                                          
                                                          #verbatimTextOutput("tree_text_GA")  # Salida para el texto del árbol
                                                          
                                                          
                                                          
                                                          
                                                 ),
                                                 
                                                 tabPanel("Gráfica de Árbol y Reglas",
                                                          selectInput("tree_run_GA", "Seleccione el número de ejecución para mostrar las reglas del arbol:", choices = 1:30),
                                                          
                                                          downloadButton("download_tree_plot_GA", "Descargar Gráfico del Árbol", class = "butt1"),  # Botón de descarga
                                                          tags$head(tags$style(".butt1{background-color:orange;} .butt{color: black;}")),
                                                          
                                                          plotOutput("tree_plot_GA", height = "600px"),  # Gráfico del árbol
                                                          
                                                          
                                                          downloadButton("download_tree_text_GA", "Descargar Árbol como Texto", class = "butt1"),
                                                          tags$head(tags$style(".butt1{background-color:orange;} .butt{color: black;}")),
                                                          
                                                          verbatimTextOutput("tree_text_GA"),  # Salida para el texto del árbol
                                                          
                                                 ),
                                                 
                                                 tabPanel("Resultados 30 veces", verbatimTextOutput("result_30_GA"),
                                                          downloadButton("download_summary_30_GA", "Descargar 30 ejecuciones", class = "butt1"),
                                                          tags$head(tags$style(".butt1{background-color:orange;} .butt{color: black;}")),
                                                          DTOutput("metrics_table_30_GA"),
                                                          
                                                          
                                                 ),
                                                 tabPanel("Gráfica de progreso de GA",
                                                          plotOutput("ga_plot")       
                                                 )
                                                 
                                                 
                                                 
                                         )
                                         
                                ),
                                tabPanel("Referencias",
                                         tabsetPanel(
                                                 tabPanel("Referencias",
                                                          h3(strong("Referencias utilizadas:")),
                                                          tags$ul(
                                                                  tags$li("Guvenir, H., Acar, B., & Muderrisoglu, H. (1998). UCI Machine Learning Repository: Arrhythmia Dataset.
                                                        UCI Machine Learning Repository. Accesado el 05 diciembre 2024, desde:",  tags$a(href = "https://archive.ics.uci.edu/ml/datasets/arrhythmia", "https://archive.ics.uci.edu/ml/datasets/arrhythmia" )),
                                                                  br(),
                                                                  
                                                                  tags$li(
                                                                          "Shiny. Shiny.rstudio.com. Accesado 05 de diciembre 2024, desde:", tags$a(href = "https://shiny.posit.co", "https://shiny.posit.co")
                                                                  ),
                                                                  br(),
                                                                  
                                                                  tags$li(
                                                                          "Paquete Fselector. Funciones para seleccionar atributos de un conjunto de datos determinado. Accesado 05 de Diciembre 2024, desde:", tags$a(href = "https://rdrr.io/cran/FSelector", "https://rdrr.io/cran/FSelector") 
                                                                          
                                                                          
                                                                  ),
                                                                  br(),
                                                                  
                                                                  tags$li(
                                                                          "R para la Computación Estadística. Página oficial para descargar o consultar paquetes de Rsudio. Accesado 05 de diciembre 2024, desde:",  tags$a(href = "https://www.r-project.org","https://www.r-project.org")
                                                                  ),
                                                                  
                                                          ),
                                                          hr(),
                                                          h3(strong("Nombre de autores:")),
                                                          
                                                          tags$ul(
                                                                  tags$li(
                                                                          strong("M. Sc. Santiago Arias García")
                                                                          
                                                                  ),
                                                                  tags$li(
                                                                          strong("Dr. José Hernández Torruco")
                                                                          
                                                                  ),
                                                                  tags$li(
                                                                          strong("Dra. Betania Hernández Ocaña")
                                                                          
                                                                  ),
                                                                  
                                                                  #img(src = "dacyti.jpg", height = "238px", width = "216px")   
                                                          )
                                                          
                                                          
                                                 )
                                                 
                                         )
                                         
                                         
                                )
                        )
                )
        )
)








# Server
server <- function(input, output, session) {
        
        # Inicialmente deshabilitar botones de descarga
        shinyjs::hide("download_data")
        shinyjs::hide("download_results_all")
        shinyjs::hide("download_data2")
        shinyjs::hide("download_results")
       # shinyjs::hide("download_trees")
        shinyjs::hide("download_summary_30")
        shinyjs::hide("download_vars_GA")
        shinyjs::hide("download_results_GA")
        shinyjs::hide("download_summary_30_GA")
        #shinyjs::hide("download_trees_GA")
        shinyjs::hide("tree_run")
        shinyjs::hide("download_tree_plot")
        shinyjs::hide("download_tree_text")
        
   
        shinyjs::hide("tree_run_GA")
        shinyjs::hide("download_tree_plot_GA")
        shinyjs::hide("download_tree_text_GA")
        


        
        
        
        
        # Dataset reactivo
        dataset <- reactiveVal(NULL)
        
        
        
        
        
        # Cargar el dataset de arrítmias al seleccionar el dataset precargado
        observeEvent(input$load_sample, {
                arritmia <- read.csv("arrhythmia.csv", header = FALSE, na.strings = "?", stringsAsFactors = FALSE)
                dataset(arritmia)
                updateTabsetPanel(session, "main_tabs", selected = "Datos")  # Cambiar a la pestaña de datos
        })
        
        # Cargar dataset seleccionado o subido
        #   observeEvent(input$data_source, {
        #          if (input$data_source == "preloaded") {
        #                 dataset(arritmia)
        #        } else {
        #               req(input$file_upload)
        #              data <- read.csv(input$file_upload$datapath, header = TRUE, sep = ",")
        #             dataset(data)
        #    }
        #})
        
        # Actualizar variables disponibles para predecir
        observe({
                data <- dataset()
                updateSelectInput(session, "target_var", choices = c("", colnames(data)), selected = "")
        })
        
        # Cargar un archivo CSV
        observeEvent(input$file1, {
                req(input$file1)
                data <- read.csv(input$file1$datapath, header = input$header, na.strings = "?", stringsAsFactors = FALSE)
                dataset(data)
                updateTabsetPanel(session, "main_tabs", selected = "Datos")  # Cambiar a la pestaña de datos
                
        })
        
        
        # Descripción y vista de los datos
        output$data_view <- renderUI({
                req(dataset())
                tags$div(
                        h5("Número de instancias y columnas"),
                        p(paste("Filas:", nrow(dataset()))),
                        p(paste("Columnas:", ncol(dataset()))),
                        hr()#,,
                        #  h5("Vista de Datos"),
                        #   DTOutput("data_table")
                )
        })
        
        
        
        # Mostrar información de la clase a predecir
        observeEvent(input$target_var, {
                req(input$target_var != "")
                data <- dataset()
                target <- data[[input$target_var]]
                class_levels <- sort(unique(target))
                
                updateSelectInput(session, "negative_class", choices = class_levels)
                updateSelectInput(session, "positive_class", choices = class_levels[-1], selected = class_levels[-1])
                
                output$class_info <- renderText({
                        paste(
                                "Frecuencia por clase:\n",
                                paste0(class_levels, ": ", table(target), collapse = "\n")
                        )
                })
                
                
        })
        
        
        
        # Actualizar opciones de clases negativas y positivas
        observe({
                neg_class <- input$negative_class
                pos_classes <- setdiff(sort(unique(dataset()[[input$target_var]])), neg_class)
                updateSelectInput(session, "positive_class", choices = pos_classes, selected = pos_classes)
                
                updateSelectInput(session, "target_var", selected = input$target_var)
                
        })
        
        # Convertir a clase binaria
        observeEvent(input$convert_binary, {
                data <- dataset()
                target <- data[[input$target_var]]
                data[[input$target_var]] <- ifelse(target == input$negative_class, 1, 2)
                dataset(data)
                showNotification("Conversión a clase binaria realizada.")
                
                # Actualizar frecuencias después de la conversión
                output$class_info <- renderText({
                        new_target <- data[[input$target_var]]
                        paste(
                                "Frecuencia por clase:\n",
                                paste0("1: ", sum(new_target == 1), "\n", "2: ", sum(new_target == 2))
                                
                        )
                        
                })
                
                
        })
        
        # Mostrar datos
        output$data_table <- renderDataTable({
                req(dataset())
                dataset()
        })
        
        
        output$summary_view <- renderUI({
                req(dataset())
                tags$div(
                        h5("Resumen Estadístico"),
                        verbatimTextOutput("summary")
                )
        })
        
        output$summary <- renderPrint({
                summary(dataset())
        })
        
        
        
        # Gráfica de NAs
        # output$na_plot <- renderPlot({
        #        data <- dataset()
        #       missing_cols <- data[, colSums(is.na(data)) > 0]
        #      plot_missing(missing_cols)
        #})
        
        
        
        #  output$na_plot <- renderPlot({
        #         data <- dataset()
        #        missing_cols <- colSums(is.na(data))
        #       if (all(missing_cols == 0)) {
        #              barplot(1, names.arg = "Sin Valores Nulos", col = "green", main = "Estado de Valores Nulos")
        #     } else {
        #            barplot(missing_cols, las = 2, col = "red", main = "Valores Nulos por Columna")
        #    }
        #})
        
        # Gráfica de valores nulos
        output$na_plot <- renderPlot({
                req(dataset())
                data <- dataset()
                missing_cols <- colSums(is.na(data))
                if (any(missing_cols > 0)) {
                        missing_data <- data[, colSums(is.na(data)) > 0, drop = FALSE]
                        plot_missing(missing_data)
                } else {
                        barplot(1, names.arg = "Sin Valores Nulos", col = "green", main = "Estado de Valores Nulos: Todos los datos completos")
                }
        })
        
        
        
        
        
        
        # Mensaje sobre columnas con muchos NAs
        output$na_message <- renderText({
                data <- dataset()
                na_cols <- colnames(data)[colSums(is.na(data)) / nrow(data) > 0.8]
                if (length(na_cols) > 0) {
                        paste("Columnas con más del 80% de datos nulos:", paste(na_cols, collapse = ", "))
                } else {
                        "No hay columnas con más del 80% de datos nulos."
                }
        })
        
        # Eliminar columnas con más del 80% de datos nulos
        observeEvent(input$remove_high_na, {
                data <- dataset()
                cols_to_remove <- colnames(data)[colSums(is.na(data)) / nrow(data) > 0.8]
                data <- data[, !(colnames(data) %in% cols_to_remove)]
                dataset(data)
                showNotification(paste("Columnas eliminadas:", paste(cols_to_remove, collapse = ", ")))
                
                #dataset(updated_data)
                dataset(data)
        })
        
        # Aplicar imputación
        observeEvent(input$apply_imputation, {
                req(dataset())  # Asegúrate de que el dataset no sea NULL
                req(input$imputation_methods)  # Asegúrate de que se haya seleccionado un método
                
                data <- dataset()  # Obtén el dataset actual
                method <- input$imputation_methods  # Método seleccionado
                
                # Identificar columnas numéricas con valores nulos
                na_cols <- colnames(data)[sapply(data, is.numeric) & colSums(is.na(data)) > 0]
                if (length(na_cols) == 0) {
                        showNotification("No hay columnas numéricas con valores nulos para imputar.", type = "message")
                        return()
                }
                
                # Aplicar imputación según el método seleccionado
                for (col in na_cols) {
                        if (method == "mean") {
                                # Imputar con la media
                                data[[col]][is.na(data[[col]])] <- round(mean(data[[col]], na.rm = TRUE))
                        } else if (method == "median") {
                                # Imputar con la mediana
                                data[[col]][is.na(data[[col]])] <-round( median(data[[col]], na.rm = TRUE))
                        } else if (method == "mode") {
                                # Imputar con la moda
                                mode_val <- as.numeric(names(sort(table(data[[col]]), decreasing = TRUE)[1]))
                                data[[col]][is.na(data[[col]])] <- round(mode_val)
                        }
                }
                
                # Actualizar el dataset reactivo
                dataset(data)
                
                # Mostrar notificación
                showNotification("Imputación aplicada con éxito.", type = "message")
                
                # Activar botones de descarga cuando los resultados estén disponibles
                shinyjs::show("download_data")
        })
        
        
        
        #para descargar el dataset
        
        output$download_data <- downloadHandler(
                filename = function() {
                        paste("dataset_imputado_", Sys.Date(), ".csv", sep = "")
                },
                content = function(file) {
                        write.csv(dataset(), file, row.names = FALSE)
                }
        )
        
        # observeEvent(input$reset_app, {
        #        dataset(NULL)
        
        
        #       showNotification("Aplicación reiniciada. Puede cargar un nuevo dataset.", type = "warning")
        #})        
        
        
        observeEvent(input$reset_app, {
                # Reiniciar el dataset reactivo
                dataset(NULL)
                
                # Reiniciar los selectores
                updateSelectInput(session, "target_var", choices = NULL, selected = NULL)
                
                # Usar NULL para limpiar las opciones y valores seleccionados de los selectores
                updateSelectInput(session, "negative_class", choices = c(""), selected = NULL)
                updateSelectInput(session, "positive_class", choices = c(""), selected = NULL)
                
                # Cambiar a la pestaña "Información"
                updateTabsetPanel(session, "main_tabs", selected = "Información")
                
                # Limpiar las salidas relacionadas con las frecuencias por clase
                output$class_info <- renderText(NULL)
                
                # Mostrar notificación
                showNotification("Aplicación reiniciada. Puede cargar un nuevo dataset.", type = "warning")
        })
        
        # clasificadores con el dataset completo-----------------------------------------
        
        # Variable reactiva para almacenar los resultados
        results_summary_all <- reactiveVal(NULL)
        
        observeEvent(input$apply_classifier_30times_all, {
                req(dataset(), input$target_var, input$classifier_all, input$positive_class)
                
                
                withProgress(message = "Ejecutando clasificador 30 veces...", value = 0, {
                        
                        # Configuración inicial
                        data <- dataset()
                        target_var <- input$target_var
                        pos_class <- input$positive_class
                        semillas <- 1:30
                        data[[target_var]] <- as.factor(data[[target_var]])
                        
                        # Almacenar métricas
                        metrics_all <- data.frame(
                                accuracy = numeric(length(semillas)),
                                sensitivity = numeric(length(semillas)),
                                specificity = numeric(length(semillas)),
                                balanced_accuracy = numeric(length(semillas)),
                                kappa = numeric(length(semillas)),
                                auc = numeric(length(semillas))
                        )
                        
                        # Ejecución del modelo 30 veces con diferentes semillas
                        for (i in semillas) {
                                set.seed(i)
                                train_index <- createDataPartition(data[[target_var]], p = 2/3, list = FALSE)
                                train_data <- data[train_index, ]
                                test_data <- data[-train_index, ]
                                
                                # Entrenar modelo
                                model_all <- switch(input$classifier,
                                                    "C5.0" = C5.0(as.formula(paste(target_var, "~ .")), data = train_data),
                                                    "Rpart" = rpart(as.formula(paste(target_var, "~ .")), data = train_data),
                                                    "PART" = PART(as.formula(paste(target_var, "~ .")), data = train_data),
                                                    "JRip" = JRip(as.formula(paste(target_var, "~ .")), data = train_data),
                                                    "OneR" = OneR(as.formula(paste(target_var, "~ .")), data = train_data),
                                                    "SVMlin" = svm(as.formula(paste(target_var, "~ .")), data = train_data),
                                                    "KNN" = knn3(as.formula(paste(target_var, "~ .")), data = train_data),
                                                    "Random Forest" = randomForest(as.formula(paste(target_var, "~ .")), data = train_data)
                                )
                                
                                # Predicciones y matriz de confusión
                                pred <- predict(model_all, newdata = test_data, type = "class")
                                cm <- confusionMatrix(pred, test_data[[target_var]], positive = pos_class)
                                
                                # Guardar métricas
                                metrics_all$balanced_accuracy[i] <- cm$byClass["Balanced Accuracy"]
                                metrics_all$sensitivity[i] <- cm$byClass["Sensitivity"]
                                metrics_all$specificity[i] <- cm$byClass["Specificity"]
                                metrics_all$accuracy[i] <- cm$overall["Accuracy"]
                                # AUC
                                roc_res <- roc(as.numeric(test_data[[target_var]]), as.numeric(pred), auc = TRUE)
                                metrics_all$auc[i] <- roc_res$auc
                                
                                metrics_all$kappa[i] <- cm$overall["Kappa"]
                                # Actualizar progreso
                                incProgress(1 / length(semillas))
                                
                        }
                        
                        # Resumen de métricas
                        summary_metrics_all <- data.frame(
                                Statistic = c("Mean", "Min", "Max", "SD"),
                                Balanced_Accuracy = c(mean(metrics_all$balanced_accuracy, na.rm = TRUE),
                                                      min(metrics_all$balanced_accuracy, na.rm = TRUE),
                                                      max(metrics_all$balanced_accuracy, na.rm = TRUE),
                                                      sd(metrics_all$balanced_accuracy, na.rm = TRUE)),
                                Sensitivity = c(mean(metrics_all$sensitivity, na.rm = TRUE),
                                                min(metrics_all$sensitivity, na.rm = TRUE),
                                                max(metrics_all$sensitivity, na.rm = TRUE),
                                                sd(metrics_all$sensitivity, na.rm = TRUE)),
                                Specificity = c(mean(metrics_all$specificity, na.rm = TRUE),
                                                min(metrics_all$specificity, na.rm = TRUE),
                                                max(metrics_all$specificity, na.rm = TRUE),
                                                sd(metrics_all$specificity, na.rm = TRUE)),
                                Accuracy = c(mean(metrics_all$accuracy, na.rm = TRUE),
                                             min(metrics_all$accuracy, na.rm = TRUE),
                                             max(metrics_all$accuracy, na.rm = TRUE),
                                             sd(metrics_all$accuracy, na.rm = TRUE)),
                                AUC = c(mean(metrics_all$auc, na.rm = TRUE),
                                        min(metrics_all$auc, na.rm = TRUE),
                                        max(metrics_all$auc, na.rm = TRUE),
                                        sd(metrics_all$auc, na.rm = TRUE)),
                                Kappa = c(mean(metrics_all$kappa, na.rm = TRUE),
                                          min(metrics_all$kappa, na.rm = TRUE),
                                          max(metrics_all$kappa, na.rm = TRUE),
                                          sd(metrics_all$kappa, na.rm = TRUE))
                                
                        )
                        
                        # Redondear a 4 dígitos todas las columnas excepto "Statistic"
                        summary_metrics_all[-1] <- lapply(summary_metrics_all[-1], function(x) round(x, 4))
                        
                        # Mostrar resultados en el UI
                        output$metrics_table_all <- DT::renderDataTable({
                                datatable(summary_metrics_all)
                        })
                        
                        
                        
                        
                        # Almacenar en la reactiva
                        results_summary_all(summary_metrics_all)
                        
                        
                        #output$metrics_summary <- renderPrint({
                        
                        #       summary_metrics
                        #})
                        
                        showNotification("Clasificador ejecutado 30 veces y métricas calculadas.", type = "message")
                        
                        shinyjs::show("download_results_all")
                        
                })
        })
        
        # Descargar los resultados
        output$download_results_all <- downloadHandler(
                filename = function() {
                        paste("resultados_metricas_All_", Sys.Date(), ".csv", sep = "")
                },
                content = function(file) {
                        write.csv(results_summary_all(), file, row.names = FALSE)
                }
        )
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        #---------aplicacion de filtro##################################################
        
        
        # Reactive para almacenar el dataset filtrado
        filtered_dataset <- reactiveVal(NULL)
        
        # Función para aplicar el filtro
        observeEvent(input$apply_filter, {
                req(dataset(), input$target_var, input$filter_method)
                
                
                withProgress(message = "Aplicando filtro de variables...", value = 0, {
                        # Paso 1: Obtener el dataset cargado
                        incProgress(0.2, detail = "Preparando datos...")
                        
                        
                        # Obtener el dataset imputado
                        data <- dataset()
                        target_var <- input$target_var
                        
                        # Verificar que la variable seleccionada exista y convertirla en factor
                        if (!(target_var %in% colnames(data))) {
                                showNotification("La variable seleccionada no existe en el dataset.", type = "error")
                                return()
                        }
                        data[[target_var]] <- as.factor(data[[target_var]])
                        
                        # Definir las variables predictoras
                        target <- data[[target_var]]
                        predictors <- data[, setdiff(names(data), target_var)]
                        
                        # Aplicar el filtro según el método seleccionado
                        filter_method <- switch(input$filter_method,
                                                "CFS" = cfs,
                                                "Consistency" = consistency,
                                                "Chi.squared" = chi.squared,
                                                "Information Gain" = information.gain,
                                                "Gain Ratio" = gain.ratio,
                                                "Symmetrical Uncertainty" = symmetrical.uncertainty,
                                                "OneR" = oneR)
                        
                        filtered <- filter_method(target ~ ., predictors)
                        
                        
                        # Mostrar el número de variables seleccionadas
                        output$num_selected_vars <- renderText({
                                paste("Número de variables seleccionadas:", length(filtered), collapse = "\n")
                        })
                        
                        
                        # Obtener las variables seleccionadas
                        incProgress(0.7, detail = "Filtrando variables seleccionadas...")
                        # Variables seleccionadas por el filtro
                        #selected_vars <- names(filtered)[filtered > 0]
                        #filtered_data <- data[, c(selected_vars, target_var)]
                        
                        #filtered_data <- data[, c(filtered, target_var)]
                        
                        # Actualizar el dataset con las variables seleccionadas
                        #dataset(filtered_data)
                        
                        # Variables seleccionadas y dataset filtrado
                        #selected_vars <- names(filtered)[filtered > 0]
                        filtered_data <- data[, c(filtered, target_var), drop = FALSE]
                        
                        # Guardar el dataset filtrado
                        filtered_dataset(filtered_data)
                        
                        
                        # Mostrar el resultado del filtro
                        # print("Resultado del filtro:")
                        # print(filtered_dataset)#, 
                        # print(filtered_data)
                        
                        # Actualizar el progreso
                        incProgress(0.9, detail = "Finalizando...")
                        Sys.sleep(0.5)  # Simulación de tiempo de cálculo
                        
                        # Guardar el número de variables seleccionadas en una variable reactiva
                        num_selected_vars(filtered)
                        
                })
                # Notificación de éxito
                showNotification("Filtro aplicado con éxito.", type = "message")
                #filtered_data)
                
                shinyjs::show("download_data2")
        })
        
        
        
        # Reactiva para almacenar el número de variables seleccionadas
        num_selected_vars <- reactiveVal(0)
        
        
        
        # Mostrar un resumen de las variables seleccionadas
        output$summary_table <- DT::renderDataTable({
                req(filtered_dataset())  # Asegúrate de que el dataset no esté vacío
                datatable(filtered_dataset())  # Usando el dataset filtrado
                
        })
        
        
        
        #arritmia.C45<- select(arritmia, all_of(weights), V280)
        
        # arritmia.C45$V280 <- factor(c(arritmia.C45$V280))
        
        #para descargar el dataset
        
        output$download_data2 <- downloadHandler(
                filename = function() {
                        paste("dataset_filtrado_", Sys.Date(), ".csv", sep = "")
                },
                content = function(file) {
                        write.csv(filtered_dataset(), file, row.names = FALSE)
                }
        )
        
        
        
        
        # Renderizar el gráfico de correlación
        output$correlation_plot <- renderPlot({
                req(filtered_dataset())  # Asegúrate de que el dataset no esté vacío
                
                # Crear un gráfico de correlación (por ejemplo, utilizando corrplot)
                cor_matrix <- cor(filtered_dataset()[, sapply(filtered_dataset(), is.numeric)], use = "complete.obs")
                corrplot::corrplot(cor_matrix, method = "circle")
        })
        
        #Renderizar el resumen del modelo
        output$model_summary <- renderPrint({
                summary(filtered_dataset())
                #  req(dataset())  # Asegúrate de que el dataset no esté vacío
                
                # Aquí puedes agregar el modelo de clasificación o resumen que deseas#
        })
        
        
        # Mostrar la variable objetivo seleccionada
        output$selected_target <- renderText({
                req(input$target_var)
                paste(" ", input$target_var)
        })
        
        # Resetear selección
        # observeEvent(input$reset_filter, {
        #selected_variables(NULL)
        #dataset(NULL)
        #updateSelectInput(session = renderDataTable(NULL))
        #updateSelectInput(session, "filter_method", selected = NULL)
        #updateSelectInput(session, "classifier", selected = NULL)
        #        showNotification("Selección de variables reiniciada.", type = "warning")
        #})
        
        
        
        
        
        #------------------aplicacion del clasificador 30 veces##########################################################
        # Variable reactiva para almacenar los resultados
        results_summary <- reactiveVal(NULL)
        results_summary_30 <- reactiveVal(NULL)
        
        observeEvent(input$apply_classifier_30times, {
                req(filtered_dataset(), input$target_var, input$classifier, input$positive_class)
                
                withProgress(message = "Ejecutando clasificador 30 veces...", value = 0, {
                        
                        # Configuración inicial
                        data <- filtered_dataset()
                        target_var <- input$target_var
                        pos_class <- input$positive_class
                        semillas <- 1:30
                        
                        # Almacenar métricas
                        metrics <- data.frame(
                                balanced_accuracy = numeric(length(semillas)),
                                sensitivity = numeric(length(semillas)),
                                specificity = numeric(length(semillas)),
                                accuracy = numeric(length(semillas)),
                                auc = numeric(length(semillas)),
                                kappa = numeric(length(semillas))
                                
                        )
                        
                        models <- list()  # Lista para guardar los árboles
                        
                        # Ejecución del modelo 30 veces con diferentes semillas
                        for (i in semillas) {
                                set.seed(i)
                                train_index <- createDataPartition(data[[target_var]], p = 2/3, list = FALSE)
                                train_data <- data[train_index, ]
                                test_data <- data[-train_index, ]
                                
                                # Entrenar modelo
                                model <- switch(input$classifier,
                                                "C5.0" = C5.0(as.formula(paste(target_var, "~ .")), data = train_data),
                                                "Rpart" = rpart(as.formula(paste(target_var, "~ .")), data = train_data),
                                                "PART" = PART(as.formula(paste(target_var, "~ .")), data = train_data),
                                                "JRip" = JRip(as.formula(paste(target_var, "~ .")), data = train_data),
                                                "OneR" = OneR(as.formula(paste(target_var, "~ .")), data = train_data),
                                                "SVMlin" = svm(as.formula(paste(target_var, "~ .")), data = train_data),
                                                "KNN" = knn3(as.formula(paste(target_var, "~ .")), data = train_data),
                                                "Random Forest" = randomForest(as.formula(paste(target_var, "~ .")), data = train_data)
                                )
                                
                                
                                #models[[i]] <- model  # Guardar el modelo
                                
                                # Dentro del bucle de ejecución de modelos
                                if (input$classifier == "C5.0") {
                                        models[[i]] <- model  # Guardar el modelo C5.0
                                } else {
                                        models[[i]] <- NULL  # Para otros clasificadores, no es necesario guardar
                                }
                                
                                
                                
                                
                                # Predicciones y matriz de confusión
                                pred <- predict(model, newdata = test_data, type = "class")
                                cm <- confusionMatrix(pred, test_data[[target_var]], positive = pos_class)
                                
                                # Guardar métricas
                                metrics$balanced_accuracy[i] <- cm$byClass["Balanced Accuracy"]
                                metrics$sensitivity[i] <- cm$byClass["Sensitivity"]
                                metrics$specificity[i] <- cm$byClass["Specificity"]
                                metrics$accuracy[i] <- cm$overall["Accuracy"]
                                # AUC
                                roc_res <- roc(as.numeric(test_data[[target_var]]), as.numeric(pred), auc = TRUE)
                                metrics$auc[i] <- roc_res$auc
                                
                                metrics$kappa[i] <- cm$overall["Kappa"]
                                # Actualizar progreso
                                incProgress(1 / length(semillas))
                                
                        }
                        
                        # Resumen de métricas
                        summary_metrics <- data.frame(
                                Statistic = c("Mean", "Min", "Max", "SD"),
                                Balanced_Accuracy = c(mean(metrics$balanced_accuracy, na.rm = TRUE),
                                                      min(metrics$balanced_accuracy, na.rm = TRUE),
                                                      max(metrics$balanced_accuracy, na.rm = TRUE),
                                                      sd(metrics$balanced_accuracy, na.rm = TRUE)),
                                Sensitivity = c(mean(metrics$sensitivity, na.rm = TRUE),
                                                min(metrics$sensitivity, na.rm = TRUE),
                                                max(metrics$sensitivity, na.rm = TRUE),
                                                sd(metrics$sensitivity, na.rm = TRUE)),
                                Specificity = c(mean(metrics$specificity, na.rm = TRUE),
                                                min(metrics$specificity, na.rm = TRUE),
                                                max(metrics$specificity, na.rm = TRUE),
                                                sd(metrics$specificity, na.rm = TRUE)),
                                Accuracy = c(mean(metrics$accuracy, na.rm = TRUE),
                                             min(metrics$accuracy, na.rm = TRUE),
                                             max(metrics$accuracy, na.rm = TRUE),
                                             sd(metrics$accuracy, na.rm = TRUE)),
                                AUC = c(mean(metrics$auc, na.rm = TRUE),
                                        min(metrics$auc, na.rm = TRUE),
                                        max(metrics$auc, na.rm = TRUE),
                                        sd(metrics$auc, na.rm = TRUE)),
                                Kappa = c(mean(metrics$kappa, na.rm = TRUE),
                                          min(metrics$kappa, na.rm = TRUE),
                                          max(metrics$kappa, na.rm = TRUE),
                                          sd(metrics$kappa, na.rm = TRUE))
                                
                        )
                        
                        # Redondear a 4 dígitos todas las columnas excepto "Statistic"
                        summary_metrics[-1] <- lapply(summary_metrics[-1], function(x) round(x, 4))
                        
                        # Guardar los modelos y las métricas en results_summary
                        results_summary_30(list(metrics = metrics, models = models))
                        
                        # Mostrar resultados en el UI
                        output$metrics_table <- DT::renderDataTable({
                                datatable(summary_metrics)
                                
                                
                                
                        })
                        
                        
                        
                        ## Mostrar los resultados en la tabla
                        # output$metrics_table_30 <- DT::renderDataTable({
                        #       req(results_summary_30())
                        #      datatable(results_summary_30()$metrics)
                        
                        
                        # })
                        
                        
                        
                        
                        # Mostrar los resultados en la tabla
                        output$metrics_table_30 <- DT::renderDataTable({
                                req(results_summary_30())
                                
                                # Redondear las columnas numéricas de las métricas a 4 dígitos
                                metrics_rounded <- results_summary_30()$metrics
                                metrics_rounded[] <- lapply(metrics_rounded, function(col) {
                                        if (is.numeric(col)) {
                                                round(col, 4)  # Redondear si es numérico
                                        } else {
                                                col  # Dejar sin cambios si no es numérico
                                        }
                                })
                                
                                # Mostrar la tabla con los valores redondeados
                                datatable(metrics_rounded, options = list(pageLength = 50))
                                
                                
                                
                                
                        })
                        
                        
                        
                        # Almacenar en la reactiva
                        results_summary(summary_metrics)
                        
                        
                        #output$metrics_summary <- renderPrint({
                        
                        #       summary_metrics
                        #})
                        
                        showNotification("Clasificador ejecutado 30 veces y métricas calculadas.", type = "message")
                        
                })
                
                shinyjs::show("download_results")
                
                shinyjs::show("download_summary_30")
                
               # shinyjs::show("download_trees")
                
                shinyjs::show("tree_run")
                
                shinyjs::show("download_tree_text")
                
                shinyjs::show("download_tree_plot")
                
                
                
        })
        
        #para graficar un modelo de arbol de c4.5
        
       # output$tree_plot <- renderPlot({
        #        req(results_summary_30()$models, input$tree_run)
                
                # Obtener el modelo seleccionado
         #       selected_model <- results_summary_30()$models[[as.numeric(input$tree_run)]]
                
                # Convertirlo a formato partykit para graficarlo
          #      tree_party <- as.party(selected_model)
           #     plot(tree_party, main = paste("Árbol de la ejecución", input$tree_run))
                
               # shinyjs::show("download_trees")
        #})
        
        
       
        
        
        
        # Descargar los resultados del promedio de las métricas
        output$download_results <- downloadHandler(
                filename = function() {
                        paste("resultados_metricas_filtro_", Sys.Date(), ".csv", sep = "")
                },
                content = function(file) {
                        write.csv(results_summary(), file, row.names = FALSE)
                }
        )
        
        
        
        
        observeEvent(input$tree_run, {
                selected_model <- results_summary_30()$models[[as.numeric(input$tree_run)]]
                print(selected_model)  # Verifica si el modelo tiene una estructura válida
        })
        
        

        
        ### para descargar los arboles 30 
        
     #   output$download_trees <- downloadHandler(
      #          filename = function() {
       #                 paste("trees_", Sys.Date(), ".pdf", sep = "")
        #        },
         #       content = function(file) {
          #              req(results_summary_30()$models)
           #             
            #            pdf(file, width = 35, height = 20)
             #           for (i in seq_along(results_summary_30()$models)) {
              #                  tree_party <- as.party(results_summary_30()$models[[i]])
               #                 plot(tree_party, main = paste("Árbol de la ejecución", i))
                #        }
                 #       dev.off()
                #}
        #)
        
        # Graficar el árbol del modelo seleccionado222
        
        output$tree_plot <- renderPlot({
                req(results_summary_30()$models, input$tree_run)
                
                # Obtén el modelo seleccionado
                selected_model <- results_summary_30()$models[[as.numeric(input$tree_run)]]
                
                # Verifica que el modelo sea de tipo C5.0 antes de intentar graficarlo
                if (!inherits(selected_model, "C5.0")) {
                        stop("El modelo seleccionado no es un árbol de decisión C5.0.")
                }
                
                # Graficar directamente usando plot.C5.0
                plot(selected_model, main = paste("Árbol de la ejecución", input$tree_run))
        })
        
        #descargar grafica de arbol seleccionado
        
        output$download_tree_plot <- downloadHandler(
                filename = function() {
                        paste0("tree_plot_execution_", input$tree_run, ".pdf")
                },
                content = function(file) {
                        req(results_summary_30()$models, input$tree_run)
                        selected_model <- results_summary_30()$models[[as.numeric(input$tree_run)]]
                        
                        # Verificar que el modelo tenga una estructura válida
                        if (!inherits(selected_model, "C5.0")) {
                                stop("El modelo seleccionado no es un árbol de decisión C5.0.")
                        }
                        
                        # Graficar el árbol
                        pdf(file, width = 65, height = 25)
                        tryCatch({
                                plot(selected_model, main = paste("Árbol de la ejecución", input$tree_run))
                        }, error = function(e) {
                                plot.new()  # Generar un gráfico vacío si hay un error
                                text(0.5, 0.5, "Error al graficar el árbol. Verifica el modelo.")
                        })
                        dev.off()
                }
        )
        
        
        
        
        
        # Mostrar la estructura del árbol como texto
        output$tree_text <- renderPrint({
                req(results_summary_30()$models, input$tree_run)
                
                # Obtener el modelo seleccionado
                selected_model <- results_summary_30()$models[[as.numeric(input$tree_run)]]
                
                # Generar un encabezado con el número del árbol
                cat("Reglas del Árbol de la ejecución: ", input$tree_run, "\n\n")
                
                # Extraer y mostrar las reglas del árbol
                tree_rules <- capture.output(summary(selected_model))
                cat(tree_rules, sep = "\n")
        })
        
    
        
        
        
        
        
        # Descargar la estructura del árbol como texto
        output$download_tree_text <- downloadHandler(
                filename = function() {
                        paste0("tree_rules_execution_", input$tree_run, ".txt")
                },
                content = function(file) {
                        req(results_summary_30()$models, input$tree_run)
                        selected_model <- results_summary_30()$models[[as.numeric(input$tree_run)]]
                        
                        # Verificar que sea un modelo C5.0
                        if (!inherits(selected_model, "C5.0")) {
                                stop("El modelo seleccionado no es un árbol de decisión C5.0.")
                        }
                        
                        # Extraer las reglas
                        tree_summary <- tryCatch({
                                summary(selected_model)$output  # Obtener reglas del árbol
                        }, error = function(e) {
                                NULL
                        })
                        
                        # Validar si hay reglas generadas
                        if (is.null(tree_summary) || length(tree_summary) == 0) {
                                tree_summary <- "No se encontraron reglas para este modelo. Verifica los datos de entrenamiento."
                        }
                        
                        # Guardar las reglas en el archivo
                        writeLines(tree_summary, file)
                }
        )
        
        
        
        
        
        
        
        
        
        
        # Descargar los resultados de las 30 ejecuciones
        output$download_summary_30 <- downloadHandler(
                filename = function() {
                        paste("30_resultados_", Sys.Date(), ".csv", sep = "")
                },
                content = function(file) {
                        write.csv(results_summary_30()$metrics, file, row.names = TRUE)
                }
        )
        
        observe({
                print(filtered_dataset())
        })
        
        
        #implementación dela lgoritmo GA -------------------------------------------------
        # Ejecutar el algoritmo genético
        
        # Reactive para almacenar el dataset filtrado
        filtered_dataset_GA <- reactiveVal(NULL)
        
        ga_result <- eventReactive(input$run_ga, {
                req(filtered_dataset(), input$positive_class)
                
                data <- filtered_dataset()
                target_var <- input$target_var
                pos_class <- input$positive_class
                
                # Validar target_var
                if (!target_var %in% colnames(data)) stop("La variable objetivo no está en el dataset.")
                
                # Crear matriz de predictores y vector objetivo
                x <- model.matrix(as.formula(paste(target_var, "~ .")), data = data)[, -1]
                y <- data[[target_var]]
                
                # Eliminar columnas constantes antes de normalizar
                x <- x[, apply(x, 2, function(col) sd(col) > 0), drop = FALSE]
                x <- scale(x, center = FALSE, scale = apply(x, 2, max))
                
                if (ncol(x) == 0) stop("No hay variables predictoras válidas.")
                
                withProgress(message = "Ejecutando algoritmo genético...", value = 0, {
                        fitness <- function(string) {
                                inc <- which(string == 1)
                                if (length(inc) == 0) {
                                        cat("No se seleccionaron variables en esta iteración.\n")
                                        return(0)
                                }
                                
                                X <- x[, inc, drop = FALSE]
                                accdata <- 0
                                semillas <- 1:30
                                
                                for (j in semillas) {
                                        set.seed(j)
                                        indxTrain <- createDataPartition(y, p = 2/3, list = FALSE)
                                        train <- X[indxTrain, , drop = FALSE]
                                        test <- X[-indxTrain, , drop = FALSE]
                                        y_train <- y[indxTrain]
                                        y_test <- y[-indxTrain]
                                        
                                        # Validar clases en y_train y y_test
                                        #if (length(unique(y_train)) < 2 || length(unique(y_test)) < 2) {
                                         #       cat("Clases insuficientes en y_train o y_test en semilla", j, "\n")
                                          #      return(0)
                                        #}
                                        
                                        # Convertir y_train e y_test a factores
                                        y_train <- as.factor(y_train)
                                        y_test <- as.factor(y_test)
                                        
                                        # Entrenar el modelo C5.0
                                        mod <- tryCatch({
                                                C5.0(as.data.frame(train), y_train)
                                        }, error = function(e) {
                                                cat("Error al entrenar el modelo en semilla", j, ":", e$message, "\n")
                                                return(NULL)
                                        })
                                        if (is.null(mod)) {
                                                next  # Saltar esta semilla si hay error
                                        }
                                        
                                        
                                        # Generar predicciones
                                        predictions <- tryCatch({
                                                predict(mod, newdata = as.data.frame(test), type = "class")
                                        }, error = function(e) {
                                                cat("Error al predecir en semilla", j, ":", e$message, "\n")
                                                return(NULL)
                                        })
                                        if (is.null(predictions)) {
                                                next  # Saltar esta semilla si hay error
                                        }
                                        
                                        
                                        # Calcular matriz de confusión
                                        cm <- confusionMatrix(predictions, y_test, positive = pos_class)
                                        accdata <- accdata + cm$byClass["Balanced Accuracy"]
                                        
                                        # Actualizar progreso
                                        incProgress(1 / (30 * input$maxiter), detail = paste("Semilla:", j))
                                }
                                
                                return(accdata / 30)
                        }
                        
                        
                        GA <- ga(
                                type = "binary",
                                fitness = fitness,
                                nBits = ncol(x),
                                popSize = input$popSize,
                                pcrossover = input$pcrossover,
                                pmutation = input$pmutation,
                                elitism = input$elitism,
                                maxiter = input$maxiter,
                                monitor = plot
                        )
                        
                        selected_vars <- colnames(x)[GA@solution[1, ] == 1]
                        if (length(selected_vars) == 0) stop("No se seleccionaron variables.")
                        
                        output$num_selected_vars_GA <- renderText({
                                paste("Número de variables seleccionadas GA:", length(selected_vars))
                        })
                        
                        selected_dataset_GA <- data[, c(selected_vars, target_var), drop = FALSE]
                        filtered_dataset_GA(selected_dataset_GA)
                        
                        incProgress(1, detail = "Finalizando...")
                        
                        list(ga = GA, selected_vars = selected_vars, selected_dataset_GA = selected_dataset_GA)
                        
                        
                })
                
                 
        })
        
        shinyjs::show("download_vars_GA")
  
        
        
        # Mostrar las variables seleccionadas en el dataset filtrado
        output$selected_vars_table_GA <- renderDT({
                req(ga_result())  # Asegúrate de que ga_result tiene resultados
                datatable(ga_result()$selected_dataset_GA, options = list(pageLength = 10))
        })
        
        
        
        # Descargar el subconjunto del dataset seleccionado
        output$download_vars_GA <- downloadHandler(
                filename = function() {
                        paste("subconjunto_seleccionado_GA_", Sys.Date(), ".csv", sep = "")
                },
                content = function(file) {
                        write.csv(ga_result()$selected_dataset_GA, file, row.names = FALSE)
                }
                
        )
        
        # Mostrar gráfica del GA
        output$ga_plot <- renderPlot({
                req(ga_result())
                plot(ga_result()$ga, main = "Progreso del Algoritmo Genético")
        })
        
        
        
        
        
        
        #----------------------implementacion de clasificadores GA
        
        
        # Variable reactiva para almacenar los resultados
        results_summary_GA <- reactiveVal(NULL)
        results_summary_30_GA <- reactiveVal(NULL)
        
        observeEvent(input$apply_classifier_30times_GA, {
                req(filtered_dataset_GA(), input$target_var, input$classifier_GA, input$positive_class)
                
                
                withProgress(message = "Ejecutando clasificador 30 veces...", value = 0, {
                        
                        # Configuración inicial
                        data <- filtered_dataset_GA()
                        target_var <- input$target_var
                        pos_class <- input$positive_class
                        semillas <- 1:30
                        data[[target_var]] <- as.factor(data[[target_var]])
                        
                        # Almacenar métricas
                        metrics_GA <- data.frame(
                                balanced_accuracy = numeric(length(semillas)),
                                sensitivity = numeric(length(semillas)),
                                specificity = numeric(length(semillas)),
                                accuracy = numeric(length(semillas)),
                                auc = numeric(length(semillas)),
                                kappa = numeric(length(semillas))
                                
                        )
                        
                        models_GA <- list()  # Lista para guardar los árboles
                        # Ejecución del modelo 30 veces con diferentes semillas
                        for (i in semillas) {
                                set.seed(i)
                                train_index <- createDataPartition(data[[target_var]], p = 2/3, list = FALSE)
                                train_data <- data[train_index, ]
                                test_data <- data[-train_index, ]
                                
                                # Entrenar modelo
                                model_GA <- switch(input$classifier_GA,
                                                   "C5.0" = C5.0(as.formula(paste(target_var, "~ .")), data = train_data),
                                                   "Rpart" = rpart(as.formula(paste(target_var, "~ .")), data = train_data),
                                                   "PART" = PART(as.formula(paste(target_var, "~ .")), data = train_data),
                                                   "JRip" = JRip(as.formula(paste(target_var, "~ .")), data = train_data),
                                                   "OneR" = OneR(as.formula(paste(target_var, "~ .")), data = train_data),
                                                   "SVMlin" = svm(as.formula(paste(target_var, "~ .")), data = train_data),
                                                   "KNN" = knn3(as.formula(paste(target_var, "~ .")), data = train_data),
                                                   "Random Forest" = randomForest(as.formula(paste(target_var, "~ .")), data = train_data)
                                )
                                
                               # models_GA[[i]] <- model_GA  # Guardar el modelo
                                
                                # Dentro del bucle de ejecución de modelos
                                if (input$classifier_GA == "C5.0") {
                                        models_GA[[i]] <- model_GA  # Guardar el modelo C5.0
                                } else {
                                        models_GA[[i]] <- NULL  # Para otros clasificadores, no es necesario guardar
                                }
                                
                                
                                # Predicciones y matriz de confusión
                                pred <- predict(model_GA, newdata = test_data, type = "class")
                                cm <- confusionMatrix(pred, test_data[[target_var]], positive = pos_class)
                                
                                # Guardar métricas
                                metrics_GA$balanced_accuracy[i] <- cm$byClass["Balanced Accuracy"]
                                metrics_GA$sensitivity[i] <- cm$byClass["Sensitivity"]
                                metrics_GA$specificity[i] <- cm$byClass["Specificity"]
                                metrics_GA$accuracy[i] <- cm$overall["Accuracy"]
                                # AUC
                                roc_res <- roc(as.numeric(test_data[[target_var]]), as.numeric(pred), auc = TRUE)
                                metrics_GA$auc[i] <- roc_res$auc
                                
                                metrics_GA$kappa[i] <- cm$overall["Kappa"]
                                # Actualizar progreso
                                incProgress(1 / length(semillas))
                                
                        }
                        
                        # Resumen de métricas
                        summary_metrics_GA <- data.frame(
                                Statistic = c("Mean", "Min", "Max", "SD"),
                                Balanced_Accuracy = c(mean(metrics_GA$balanced_accuracy, na.rm = TRUE),
                                                      min(metrics_GA$balanced_accuracy, na.rm = TRUE),
                                                      max(metrics_GA$balanced_accuracy, na.rm = TRUE),
                                                      sd(metrics_GA$balanced_accuracy, na.rm = TRUE)),
                                Sensitivity = c(mean(metrics_GA$sensitivity, na.rm = TRUE),
                                                min(metrics_GA$sensitivity, na.rm = TRUE),
                                                max(metrics_GA$sensitivity, na.rm = TRUE),
                                                sd(metrics_GA$sensitivity, na.rm = TRUE)),
                                Specificity = c(mean(metrics_GA$specificity, na.rm = TRUE),
                                                min(metrics_GA$specificity, na.rm = TRUE),
                                                max(metrics_GA$specificity, na.rm = TRUE),
                                                sd(metrics_GA$specificity, na.rm = TRUE)),
                                Accuracy = c(mean(metrics_GA$accuracy, na.rm = TRUE),
                                             min(metrics_GA$accuracy, na.rm = TRUE),
                                             max(metrics_GA$accuracy, na.rm = TRUE),
                                             sd(metrics_GA$accuracy, na.rm = TRUE)),
                                AUC = c(mean(metrics_GA$auc, na.rm = TRUE),
                                        min(metrics_GA$auc, na.rm = TRUE),
                                        max(metrics_GA$auc, na.rm = TRUE),
                                        sd(metrics_GA$auc, na.rm = TRUE)),
                                Kappa = c(mean(metrics_GA$kappa, na.rm = TRUE),
                                          min(metrics_GA$kappa, na.rm = TRUE),
                                          max(metrics_GA$kappa, na.rm = TRUE),
                                          sd(metrics_GA$kappa, na.rm = TRUE))
                                
                        )
                        
                        # Redondear a 4 dígitos todas las columnas excepto "Statistic"
                        summary_metrics_GA[-1] <- lapply(summary_metrics_GA[-1], function(x) round(x, 4))
                        
                        # Guardar los modelos y las métricas en results_summary
                        results_summary_30_GA(list(metrics = metrics_GA, models = models_GA))
                        
                        # Mostrar resultados en el UI
                        output$metrics_table_GA <- DT::renderDataTable({
                                datatable(summary_metrics_GA)
                        })
                        
                        
                        
                        # Mostrar los resultados en la tabla
                        output$metrics_table_30_GA <- DT::renderDataTable({
                                req(results_summary_30_GA())
                                
                                # Redondear las columnas numéricas de las métricas a 4 dígitos
                                metrics_rounded <- results_summary_30_GA()$metrics
                                metrics_rounded[] <- lapply(metrics_rounded, function(col) {
                                        if (is.numeric(col)) {
                                                round(col, 4)  # Redondear si es numérico
                                        } else {
                                                col  # Dejar sin cambios si no es numérico
                                        }
                                })
                                
                                # Mostrar la tabla con los valores redondeados
                                datatable(metrics_rounded, options = list(pageLength = 50))
                                
                                
                                
                                
                        })
                        
                        
                        # Almacenar en la reactiva
                        results_summary_GA(summary_metrics_GA)
                        
                        
                        #output$metrics_summary <- renderPrint({
                        
                        #       summary_metrics
                        #})
                        
                        showNotification("Clasificador ejecutado 30 veces y métricas calculadas.", type = "message")
                        
                })
                
                shinyjs::show("download_results_GA")
                
                shinyjs::show("download_summary_30_GA")
                
                shinyjs::show("download_trees_GA")
                
                shinyjs::show("tree_run_GA")
                
                shinyjs::show("download_tree_text_GA")
                
                shinyjs::show("download_tree_plot_GA")
        })
        
        #para graficar un modelo de arbol de c4.5
        
      #  output$tree_plot_GA <- renderPlot({
       #         req(results_summary_30_GA()$models, input$tree_run_GA)
        #        
         #       # Obtener el modelo seleccionado
          #      selected_model <- results_summary_30_GA()$models[[as.numeric(input$tree_run_GA)]]
           #     
            #    # Convertirlo a formato partykit para graficarlo
             #   tree_party <- as.party(selected_model)
              #  plot(tree_party, main = paste("Árbol de la ejecución", input$tree_run_GA))
                
               # shinyjs::show("download_trees_GA")
                
        #})
        
        
       
        
        
        # Descargar los resultados
        output$download_results_GA <- downloadHandler(
                filename = function() {
                        paste("resultados_metricas_GA_", Sys.Date(), ".csv", sep = "")
                },
                content = function(file) {
                        write.csv(results_summary_GA(), file, row.names = FALSE)
                }
        )
        
        
     #   output$download_trees_GA <- downloadHandler(
      #          filename = function() {
       #                 paste("trees_GA_", Sys.Date(), ".pdf", sep = "")
        #        },
         #       content = function(file) {
          #              req(results_summary_30_GA()$models)
           #             
            #            pdf(file, width = 35, height = 20)
             #           for (i in seq_along(results_summary_30_GA()$models)) {
              #                  tree_party <- as.party(results_summary_30_GA()$models[[i]])
               #                 plot(tree_party, main = paste("Árbol de la ejecución GA", i))
                #        }
                 #       dev.off()
                #}
        #)
        
        
        # Graficar el árbol del modelo seleccionado222
        
        output$tree_plot_GA <- renderPlot({
                req(results_summary_30_GA()$models, input$tree_run_GA)
                
                # Obtén el modelo seleccionado
                selected_model <- results_summary_30_GA()$models[[as.numeric(input$tree_run_GA)]]
                
                # Verifica que el modelo sea de tipo C5.0 antes de intentar graficarlo
                if (!inherits(selected_model, "C5.0")) {
                        stop("El modelo seleccionado no es un árbol de decisión C5.0.")
                }
                
                # Graficar directamente usando plot.C5.0
                plot(selected_model, main = paste("Árbol de la ejecución", input$tree_run_GA))
        })
        
        
        #descargar grafica de arbol seleccionado
        
        output$download_tree_plot_GA <- downloadHandler(
                filename = function() {
                        paste0("tree_plot_GA_execution_", input$tree_run_GA, ".pdf")  # Nombre dinámico del archivo
                },
                content = function(file) {
                        req(results_summary_30_GA()$models, input$tree_run_GA)
                        selected_model <- results_summary_30_GA()$models[[as.numeric(input$tree_run_GA)]]
                        
                        # Verifica que sea un modelo C5.0
                        if (!inherits(selected_model, "C5.0")) {
                                stop("El modelo seleccionado no es un árbol de decisión C5.0.")
                        }
                        
                        # Guardar el gráfico como archivo PDF
                        pdf(file, width = 65, height = 25)
                        plot(selected_model, main = paste("Árbol de la ejecución", input$tree_run_GA))
                        dev.off()
                }
        )
        
        
        # Mostrar la estructura del árbol como texto
        output$tree_text_GA <- renderPrint({
                req(results_summary_30_GA()$models, input$tree_run_GA)
                
                # Obtener el modelo seleccionado
                selected_model <- results_summary_30_GA()$models[[as.numeric(input$tree_run_GA)]]
                
                # Generar un encabezado con el número del árbol
                cat("Reglas del Árbol de la ejecución: ", input$tree_run_GA, "\n\n")
                
                # Extraer y mostrar las reglas del árbol
                tree_rules <- capture.output(summary(selected_model))
                cat(tree_rules, sep = "\n")
        })
        
        
        
        # Descargar la estructura del árbol como texto
        output$download_tree_text_GA <- downloadHandler(
                filename = function() {
                        paste0("tree_rules_GA_execution_", input$tree_run_GA, ".txt")  # Nombre dinámico del archivo
                },
                content = function(file) {
                        req(results_summary_30_GA()$models, input$tree_run_GA)
                        selected_model <- results_summary_30_GA()$models[[as.numeric(input$tree_run_GA)]]
                        
                        # Verifica que sea un modelo C5.0
                        if (!inherits(selected_model, "C5.0")) {
                                stop("El modelo seleccionado no es un árbol de decisión C5.0.")
                        }
                        
                        # Extrae las reglas del modelo
                        rules <- summary(selected_model)$output
                        if (is.null(rules) || length(rules) == 0) {
                                rules <- "No se encontraron reglas para este modelo."
                        }
                        
                        # Escribe las reglas en un archivo de texto
                        writeLines(rules, file)
                }
        )
        
        
        # Descargar los resultados de las 30 ejecuciones
        output$download_summary_30_GA <- downloadHandler(
                filename = function() {
                        paste("30_resultados_GA_", Sys.Date(), ".csv", sep = "")
                },
                content = function(file) {
                        write.csv(results_summary_30_GA()$metrics, file, row.names = TRUE)
                }
        )
        
        
        
        
        
}

# Correr la app
shinyApp(ui, server)