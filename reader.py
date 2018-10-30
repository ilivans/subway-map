"""
Subway network reader.
"""
import json
from collections import defaultdict

from graph import Graph


def read_json(fname):
    with open(fname, 'r') as f:
        data = json.load(f)
    return data

INTERCONNECTED_STATIONS_MOSCOW = {
    'Александровский сад': ['Александровский сад', 'Библиотека им.Ленина', 'Арбатская', 'Боровицкая'],
    'Балтийская': ['Балтийская', 'Войковская'],
    'Баррикадная': ['Баррикадная', 'Краснопресненская'],
    'Бульвар Дмитрия Донского': ['Бульвар Дмитрия Донского', 'Улица Старокачаловская'],
    'Выставочная': ['Выставочная', 'Деловой центр (Большая кольцевая линия)'],
    'Выставочный центр': ['Выставочный центр', 'ВДНХ'],
    'Деловой центр (Московское центральное кольцо)': ['Деловой центр (Московское центральное кольцо)', 'Международная'],
    'Добрынинская': ['Добрынинская', 'Серпуховская'],
    'Зябликово': ['Зябликово', 'Красногвардейская'],
    'Измайлово': ['Измайлово', 'Партизанская'],
    'Каховская': ['Каховская', 'Севастопольская'],
    'Крестьянская застава': ['Крестьянская застава', 'Пролетарская'],
    'Кузнецкий мост': ['Кузнецкий мост', 'Лубянка'],
    'Курская': ['Курская', 'Чкаловская'],
    'Локомотив': ['Локомотив', 'Черкизовская'],
    'Лужники': ['Лужники', 'Спортивная'],
    'Марксистская': ['Марксистская', 'Таганская'],
    'Менделеевская': ['Менделеевская', 'Новослободская'],
    'Новокузнецкая': ['Новокузнецкая', 'Третьяковская'],
    'Новоясеневская': ['Новоясеневская', 'Битцевский Парк'],
    'Охотный ряд': ['Охотный ряд', 'Площадь Революции', 'Театральная'],
    'Панфиловская': ['Панфиловская', 'Октябрьское поле'],
    'Петровский парк': ['Петровский парк', 'Динамо'],
    'Площадь Гагарина': ['Площадь Гагарина', 'Ленинский проспект'],
    'Площадь Ильича': ['Площадь Ильича', 'Римская'],
    'Пушкинская': ['Пушкинская', 'Тверская', 'Чеховская'],
    'Сретенский бульвар': ['Сретенский бульвар', 'Тургеневская', 'Чистые пруды'],
    'Трубная': ['Трубная', 'Цветной бульвар'],
    'Улица Милашенкова': ['Улица Милашенкова', 'Фонвизинская'],
    'Хорошевская': ['Хорошевская', 'Хорошево', 'Шелепиха', 'Полежаевская'],
}
CIRCLE_LINES_MOSCOW = ['Кольцевая', 'Московское центральное кольцо']


def read_moscow_subway(fname='data/moscow.json'):
    data = read_json(fname)
    # Prepare interconnected stations composed names
    interconnected_stations = INTERCONNECTED_STATIONS_MOSCOW.copy()
    for s, ss in interconnected_stations.copy().items():
        for s_ in ss:
            interconnected_stations[s_] = ss
    for station, neighbors in interconnected_stations.copy().items():
        interconnected_stations[station] = ' - '.join(sorted(neighbors))

    # Process multiple stations with the same name
    for line in data['lines']:
        line_name = line['name']
        stations = line['stations']
        for station in stations:
            if station['name'] == 'Деловой центр':
                station['name'] = 'Деловой центр (' + line_name + ')'

    return compose_subway_graph(data, interconnected_stations, CIRCLE_LINES_MOSCOW)


def read_spb_subway(fname='data/spb.json'):
    data = read_json(fname)
    # Prepare interconnected stations composed names
    interconnected_stations = defaultdict(lambda: [])
    for j, line0 in enumerate(data['lines']):
        for line1 in data['lines'][j + 1:]:
            stations0 = line0['stations']
            stations1 = line1['stations']
            for station0 in stations0:
                for station1 in stations1:
                    dist = (station0['lat'] - station1['lat']) ** 2 + (station0['lng'] - station1['lng']) ** 2
                    if dist < 5e-5:
                        interconnected_stations[station0['name']].append(station1['name'])
                        interconnected_stations[station1['name']].append(station0['name'])
    interconnected_stations = dict(interconnected_stations)
    for station, neighbors in interconnected_stations.items():
        interconnected_stations[station] = ' - '.join(sorted(neighbors + [station]))

    return compose_subway_graph(data, interconnected_stations, [])


def compose_subway_graph(network_data, interconnected_stations, circle_lines):
    graph = Graph()
    for line in network_data['lines']:
        line_name = line['name']
        line_color = line['hex_color']
        stations = line['stations']
        for station in stations:
            station_name = station['name']
            name = interconnected_stations.get(station_name, station_name)
            graph.add_node(name,
                           {'line_name': line_name,
                            'color': line_color if name == station_name else '000000', **station})
            station['name'] = name
        for i, from_station in enumerate(stations[:-1]):
            to_station = stations[i + 1]
            graph.add_edge(from_station['name'],
                           to_station['name'],
                           {'color': line_color, 'weight': 1})
        if line_name in circle_lines:
            graph.add_edge(stations[-1]['name'],
                           stations[0]['name'],
                           {'color': line_color, 'weight': 1})
    return graph
