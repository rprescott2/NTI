
function load_dicts() {
    var solar = new XMLHttpRequest();
    var turbine = new XMLHttpRequest();
    // var building_type = new XMLHttpRequest();
    solar.onload = function () {
        if (solar.status != 200) {
            console.log(`Ошибка ${solar.status}: ${solar.statusText}`);
        } else {
            solar_select = document.getElementById('solar')
            splitid = JSON.parse(solar.responseText)
            for (var i = 0; i < splitid.length; i++) {
                var opt = document.createElement('option');
                opt.innerHTML = splitid[i].model;
                solar_select.appendChild(opt);

            }
        };
    };

    turbine.onload = function () {
        if (turbine.status != 200) {
            console.log(`Ошибка ${turbine.status}: ${turbine.statusText}`);
        } else {
            turbine_select = document.getElementById('turbine')
            splitid = JSON.parse(turbine.responseText)
            for (var i = 0; i < splitid.length; i++) {
                var opt = document.createElement('option');
                opt.innerHTML = splitid[i].model;
                turbine_select.appendChild(opt);

            }
        };
    };

    // building_type.onload = function () {
    // if (building_type.status != 200) {
    // console.log(`Ошибка ${building_type.status}: ${building_type.statusText}`);
    // } else {
    // building_type_select = document.getElementById('building_type')
    // splitid = JSON.parse(building_type.responseText)
    // for (var i = 0; i < splitid.length; i++) {
    //             var opt = document.createElement('option');
    //             opt.innerHTML = splitid[i].name;
    //             building_type_select.appendChild(opt);

    //         }
    //     };
    // };

    solar.open('GET', 'http://127.0.0.1:8000/solar-panel/', true, "admin_nti", "12345qwe");
    solar.setRequestHeader('Content-type', 'application/json; charset=utf-8');
    solar.send()

    turbine.open('GET', 'http://127.0.0.1:8000/wind-turbine/', true, "admin_nti", "12345qwe");
    turbine.setRequestHeader('Content-type', 'application/json; charset=utf-8');
    turbine.send()

    // building_type.open('GET', 'http://172.31.63.30:8000/building-type/', true, "admin_nti", "12345qwe");
    // building_type.setRequestHeader('Content-type', 'application/json; charset=utf-8');
    // building_type.send()
}




window.onload = function () {
    load_dicts()
}


var send_request = document.getElementById("send_req");
send_request.onclick = function () {
    var send_req = new XMLHttpRequest();
    var prectot = document.getElementById('rainfall').value
    var qv2m = document.getElementById('humidity').value
    var ps = document.getElementById('pressure').value
    var t2m = document.getElementById('temperature').value
    // var file = document.getElementById('filepath').value
    var solar = document.getElementById('solar').value
    var turbine = document.getElementById('turbine').value
    var date_from = document.getElementById('date_from').value
    var date_until = document.getElementById('date_until').value
    send_req.onload = function () {
        if (send_req.status == 406) {
            alert('Недостаточно данных для прогнозирования')
        } else if (send_req.status != 200) {
            console.log(`Ошибка ${send_req.status}: ${send_req.statusText}`);
        } else {
            splitid = send_req.responseText.split(',')
            document.getElementById('wind').innerText = 'Скорость ветра (м/c): ' + splitid[3].split(':')[1].replaceAll('[', '').replaceAll(']', '')
            document.getElementById('allsky').innerText = 'Инсоляция: ' + splitid[2].split(':')[1].replaceAll('[', '').replaceAll(']', '')
            document.getElementById('kw_solar').innerText = 'Выработка с панели (КВт): ' + splitid[0].split(':')[1].replaceAll('[', '').replaceAll(']', '')
            document.getElementById('kw_wind').innerText = 'Выработка с ветряка (КВт): ' + splitid[1].split(':')[1].replaceAll('[', '').replaceAll(']', '')

        }
    };
    // 172.31.63.30
    if (date_from != '' & date_until != '') {
        var url_with_param = 'http://127.0.0.1:8000/predict/?df=' + date_from + '&du=' + date_until
    } else {
        var url_with_param = 'http://127.0.0.1:8000/predict/?prectot=' + prectot + '&qv2m=' + qv2m + '&ps=' + ps + '&t2m=' + t2m + '&solar_model=' + solar + '&turbine_model=' + turbine
    }
    send_req.open('GET', url_with_param, true, "admin_nti", "12345qwe")
    send_req.setRequestHeader('Content-type', 'application/json; charset=utf-8')
    send_req.send()
}