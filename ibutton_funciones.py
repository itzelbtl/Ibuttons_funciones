# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 00:07:14 2022

@author: Itzel
"""
# Importa librerias
import os #para manipular rutas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

def ejecutar_macro():
    def leer_archivos_ext():
        """
        Función que lee y seleciona los datos de temperatura a analizar
        Returns
        -------
        lista_archivos : list
            Lista de arrays que contienen los datos de temperatura por archivo (solo temperaturas).
        nombres_archivos : list
            Lista que contiene el nombre de los archivos.
        datos_comp : list
            Lista que contiene los datos completos de cada archivo (fecha, día, temperatura).

        """
        # Ruta de la carpeta actual
        carpeta_actual = os.path.dirname(os.path.abspath("abrir"))
        
        # Obtener la lista de archivos en la carpeta actual
        archivos = os.listdir(carpeta_actual)
        
        # Lista para almacenar los nombres de los archivos
        nombres_archivos = []
        # Lista para almacenar lo datos de los archivso
        lista_archivos=[]
        # Lista para almacenar datos completos
        datos_comp=[]
        
        # Definir la fila de inicio de los datos
        fila_in=int(input("Indica el número de fila en la que empiezan tus datos: "))
        # Definir la fila final de los datos
        fila_fin=int(input("Indica el último número de fila que desea conservar: "))
        
        # Iterar sobre los archivos
        for archivo in archivos:
            # Comprobar si el archivo es un archivo CSV
            if archivo.endswith('.csv'):
                # Construir la ruta completa del archivo
                ruta_archivo = os.path.join(carpeta_actual, archivo)
                
                # Obtener el nombre del archivo sin la extensión
                nombre_archivo = os.path.splitext(archivo)[0]
                nombres_archivos.append(nombre_archivo)
                
                # Leer el archivo CSV 
                with open(ruta_archivo, 'r') as file:
                    # Cargar los datos en una matriz de NumPy de la fil_in a la fila_fin
                    # Usar columna 2
                    matriz= np.loadtxt(file, dtype='float', skiprows=(fila_in-1), delimiter=',', 
                                       usecols=(2),max_rows=fila_fin-fila_in+1)
                    # Trasponer matriz
                    matriz=matriz.T
                    # Agregar datos a una lista
                    lista_archivos.append(matriz.tolist())
                
                # Leer el archivo CSV 
                with open(ruta_archivo, 'r') as file:
                    # Cargar los datos en una matriz de NumPy de la fil_in a la fila_fin
                    # Usar columnas 0:3
                    data_df_cru = pd.read_csv(file, skiprows=fila_in-1, usecols=range(3), 
                                              header=None, names=['Fecha_Hora', 'Letra', 'Valor'])
                    # Cargar columnas a una matriz
                    datos_cru = data_df_cru.to_numpy()
                    # Guardar en una lista
                    datos_comp.append(datos_cru.tolist())
                
        return lista_archivos, nombres_archivos,datos_comp
    
    def macro(array,no_datos_columna):
        """
        Función que genera una lista de las temperaturas ordenadas por día
        Parameters
        ----------
        array : TYPE
            DESCRIPTION.
        no_datos_columna : TYPE
            DESCRIPTION.

        Returns
        -------
        matrix : TYPE
            DESCRIPTION.
        """
        matrix = [] # Lista de matrices
        # Dividir "array" en sub-matrices de tamaño "no_datos_columna"
        while array != []:
            # Agregar una sub-matriz a "matrix" que contiene los primeros "no_datos_columna" 
            matrix.append (array [: no_datos_columna])
            # Eliminar los "no_datos_columna - 1" primeros elementos de "array"
            array = array [(no_datos_columna-1):] 
        return matrix
    
    lista_de_arrays=leer_archivos_ext() 
    # Indicar tasa de muestreo de los datos
    intervalo_pregunta=int(input("Indica la tasa de muestreo de los datos (ej: 10, 20, 30 min)\n*Solo ingresa el número*: " ))
    # Cálculo de cuantos datos hay en una columna dependiendo de la tasa de muestreo
    datos_en_col=int((1440/intervalo_pregunta)+1)
    
    # Para cada array realizar macro
    lista_macros=[] # Guardar macros en una lista
    for i in range(len(lista_de_arrays[0])):
        opera_macro=macro(lista_de_arrays[0][i],datos_en_col) #Ejectuar macro
        lista_macros.append(opera_macro) #Guardar macros en una lista
        
    # Matrices de macros
    macros2=[]
    
    for i in range(len(lista_macros)):
        lista_macros[i].pop()# Eliminar úlitmo elemento
        data_df = pd.DataFrame(np.asarray(lista_macros[i]).T) # Convertir cada lista a un array
        data_df2=pd.DataFrame(np.asarray(lista_de_arrays[2])[i]) # Extraer nombres de los archivos
        result = pd.concat([data_df2,data_df], axis=1, ignore_index=False) #Concatenar para generar archivos
        result.to_excel(lista_de_arrays[1][i]+"_macro.xlsx", index=False) # Generar excel por cada archivo
        array_macro=np.asarray(lista_macros[i])
        macros2.append(array_macro.T)
        
    print("¡Listo los archivos con las macros ejecutadas se encuentran en la carpeta!")
        
    return macros2,lista_de_arrays[1],intervalo_pregunta,lista_de_arrays

def promedio_grupal(macros):
    
    def conservar_columnas(array,cuales_filas_lista):
        # Obtener forma del array
        fila,columna=np.shape(array)
        #Lista de cuales filas conservar
        cuales_filas_array=cuales_filas_lista
        # Numero de todas las columnas que conforman el array
        columna_total=np.arange(0,columna,1)
        # no_comun: Cuales filas no hay que conservar
        no_comun=np.setdiff1d(columna_total,cuales_filas_array-1)
        # Quitar no_comun a las columnas totales
        t=np.delete(array,no_comun, axis=1)
        return t
    
    def generar_horas(incremento):
        horas = [] # Almacenar horas generadas
        for hora in range(0, 24):
            # Generar horas con el incremento
            for minuto in range(0, 60, incremento):
                # Formato ":02d" para que las horas y minutos tengan dos dígitos
                horas.append(f"{hora:02d}:{minuto:02d}")
        horas.append("24:00")
        return horas
    
    # Generar columna 0 con indices
    nums=np.arange(0,(len(macros[1])),1)
    # Generar columna 1 con el nombre de los archivos
    array_nombres=np.asarray(macros[1])
    # Generar encabezados
    encabezados = np.array([["indices", "archivos"]])
    tabla = np.column_stack((nums,array_nombres))
    # Concatenar la nueva fila de encabezados con la tabla existente.
    tabla = np.vstack((encabezados, np.column_stack((nums, array_nombres))))
    # Rango de columnas a conservar
    pregunta=str(input("Ingresa el rango de columnas a conservar como el siguiente ejemplo-> 1:8 "))
    # Arreglo de columnas a conservar
    cuales_col1=np.arange(int(pregunta[0]),int(pregunta[2])+1,1)-1  
    # Preguntar si se desean excluir columnas
    excluir=int(input("Si deseas excluir columnas ingresa 1.\nSi NO deseas excluir columnas ingresa 2: "))
    
    resultados=[]
    
    if excluir==1:
        cuantas_excluir=int(input("Ingresa cuántas columnas deseas excluir: "))
        lista_excluir=[]
        for i in range(cuantas_excluir):
            pregunta=int(input("Ingresa el número de columna que deseas excluir: "))
            lista_excluir.append(pregunta)
        # Columnas a excluir    
        lista_excluir2=np.asarray(lista_excluir)
        # Array sin las columnas a excluir
        cuales_col2=np.setdiff1d(cuales_col1,lista_excluir2)
       
        #En cada array conservar las columnas que se desean promediar
        macros_columna=[] #lista de arrays con las columnas conservadas
        for i in range(len(macros[0])):
            conservar=conservar_columnas(macros[0][i],cuales_col2)
            macros_columna.append(conservar)
        
    elif excluir==2:
        macros_columna=[] #lista de arrays con las columnas conservadas
        for i in range(len(macros[0])):
            conservar=conservar_columnas(macros[0][i],cuales_col1)
            macros_columna.append(conservar)
    
    #  Preguntar al usuario cuantos grupos tiene
    cuantos_grupos=int(input("Ingresa cuántos grupos tienes: "))
    # Almacenar nombre de grupos
    nombres_grupos=[]
    fila0=[] #Asignar fila de nombre_archivos
    for i in range(cuantos_grupos):
        print(tabla)
        print("Grupo",i+1)
        nombre_grupo=str(input("Ingresa el nombre de tu grupo: "))
        nombres_grupos.append(nombre_grupo)
        indices=str(input("Ingresa de que indice a que indice se encuentra el grupo "+ nombre_grupo+ "\nComo el siguiente ejemplo -> 0:3 "))
        
        promedios=[] # Promedio de columnas en línea base o fase exp
        for i in range(len(macros_columna)):
            r_p=np.mean(macros_columna[i],axis=1) #Promediar columnas de línea base o fase exp
            promedios.append(r_p)
        
        # Generar fila de nombres
        array_nombres_indices=(array_nombres[int(indices[0]):int(indices[2])+1]),nombre_grupo
        fila0.append(array_nombres_indices)
        lista_promedios_dias_grupos=[]
        promedios_a=np.asarray(promedios) # Array de promedio de la línea base
        promedios_dias_grupos=promedios_a[int(indices[0]):int(indices[2])+1]# Selección de promedios lb de un grupo
        lista_promedios_dias_grupos.append(promedios_dias_grupos)
        promedio_grupal=np.mean(promedios_dias_grupos, axis=0) #Promediar línea base de un grupo
        data_df = pd.DataFrame(promedios_dias_grupos.T)
        data_df2=pd.DataFrame(promedio_grupal)
        result = pd.concat([data_df,data_df2], axis=1, ignore_index=False) #Generar archivo
        resultados.append(np.asarray(result))
        
        array_horas = generar_horas(macros[2]) #Eje x
        
        plt.figure()
        plt.plot(array_horas, promedio_grupal, '-bo', markersize=3, color="black")
        plt.xticks(rotation=45, fontsize=9)
        plt.ylabel("Temperatura °C" )
        plt.xlabel("ZT")
        plt.title(nombre_grupo)
        etiquetas_x = [array_horas[i] for i in range(0, len(array_horas), 60 // macros[2])]
        plt.xticks(range(0, len(array_horas), 60 // macros[2]), etiquetas_x)
        plt.tight_layout()  
        plt.show()
    
    return(lista_promedios_dias_grupos,promedio_grupal,resultados,nombres_grupos,fila0)

def generar_archivos_promedios(promedios):
    arr=promedios[4]
    prim_fil=[]
    for i in range(len(arr)):
        # Obtener el array combinado
        combined_arr = np.concatenate((arr[i][0], [arr[i][1]]))
        prim_fil.append(combined_arr.T)
    
    original_array=promedios[2] # promedios lb o fe
    row_to_add=np.asarray(prim_fil) # fila con nombres
    #Concatenar fila de nombres y promedios
    nuevos_array=[]
    for i in range (len(original_array)):
        new_array = np.concatenate((row_to_add[i].reshape(1, len(row_to_add[i])), original_array[i]), axis=0)
        nuevos_array.append(new_array)
        df=pd.DataFrame(new_array)
        df.iloc[1:] = df.iloc[1:].astype(float)
        #Generar archivos
        df.to_excel(promedios[3][i]+"_promedio_grupal.xlsx", index=False, header=False)
    print("¡Listo los archivos promedio se encuentran en la carpeta!")
    return (nuevos_array)

def graficas_sujetos(graf_suj, macros):
    def generar_horas(incremento):
        horas = []
        for hora in range(0, 24):
            for minuto in range(0, 60, incremento):
                horas.append(f"{hora:02d}:{minuto:02d}")
        horas.append("24:00")
        return horas
    for i in range(len(graf_suj)):
        array=graf_suj[i]
        num_columns = array.shape[1]
        for column_index in range(num_columns - 1):
            # Obtener los datos de la columna
            column_data = array[1:, column_index]
            column_data=np.asarray(column_data, dtype=float)
            
            # Obtener el título del gráfico
            title = str(array[0, column_index])
            array_horas = generar_horas(macros[2])
            # Crear el gráfico
            plt.figure()
            plt.plot(array_horas,column_data,  '-bo', markersize=3, c="black")
            plt.xticks(rotation=45, fontsize=9)
            plt.title(title)
            plt.ylabel("Temperatura °C" )
            plt.xlabel("ZT")
            etiquetas_x = [array_horas[i] for i in range(0, len(array_horas), 60 // macros[2])]
            plt.xticks(range(0, len(array_horas), 60 // macros[2]), etiquetas_x)
            # Mostrar el gráficoplt.tight_layout()  # Ajustar el diseño para evitar que se recorten las etiquetas
            plt.show()
                     
def promedio_dia_noche(promedios):
    lista_concatenar_promedios=[]
    for i in range(len(promedios[2])):
        array_archivo=np.asarray(promedios[2][i]) #array de promedios
        mitad=(len(array_archivo[:,0])-1)/2
        dia_ext=int(mitad); noche_ext=int(mitad*2)
        seleccion_archivo_dia=array_archivo[0:dia_ext][:,] #datos que conforman el día
        seleccion_archivo_noche=array_archivo[dia_ext:noche_ext][:,] #datos que conforman la noche
        promedio_dia=np.mean(seleccion_archivo_dia, axis=0) # promedio del día
        promedio_noche=np.mean(seleccion_archivo_noche, axis=0) # promedio de la noche
        concatenar_promedios=np.stack((promedio_dia, promedio_noche)) #concetanera promedios en filas
        lista_concatenar_promedios.append(concatenar_promedios)
        eliminar_ult_col=np.delete(concatenar_promedios,(-1),axis=1) #eliminar columna del promedio grupal
        fig, ax = plt.subplots()
        ejey = [ concatenar_promedios[:, -1][0], concatenar_promedios[:, -1][1]]
        error=[np.std(eliminar_ult_col[0]), np.std(eliminar_ult_col[1])]
        bar_colors = ['silver', 'black']
        ax.bar([0,1],ejey, color=bar_colors,yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.scatter([0] * len(eliminar_ult_col[0]), eliminar_ult_col[0], color='black', label='Datos Barra 1')
        ax.scatter([1] * len(eliminar_ult_col[1]), eliminar_ult_col[1], color='black', label='Datos Barra 2')
        ax.set_ylabel('Temperatura °C')
        ax.set_title(promedios[3][i])
        ax.set_ylim(min(ejey)-1,max(ejey)+0.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Día", "Noche"])
        plt.show()
    print("¡Listo los archivos promedio_dia_noche se encuentran en la carpeta!")
    return (lista_concatenar_promedios)
        

def generar_archivos_promedio_dia_noche(promedios,archivo_grupal):
    arr=promedios[4]
    prim_fil=[]
    for i in range(len(arr)):
        # Obtener el array combinado
        combined_arr = np.concatenate((arr[i][0], [arr[i][1]]))
        prim_fil.append(combined_arr.T)
    
    original_array=archivo_grupal #Fila con array de promedios
    row_to_add=np.asarray(prim_fil) # Fila con nombres
    
    for i in range (len(original_array)):
        new_array = np.concatenate((row_to_add[i].reshape(1, len(row_to_add[i])), original_array[i]), axis=0)
        df=pd.DataFrame(new_array)
        df.iloc[1:] = df.iloc[1:].astype(float)
        df.to_excel(promedios[3][i]+"_promedio_dia_noche.xlsx", index=False, header=False, float_format="%.2f" )


